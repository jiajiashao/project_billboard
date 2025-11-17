def process_clip_owl(
    clip_cfg: _Dict,
    context: _Dict,
    model,
    processor,
    run_dir: Path,
    git_commit: str,
    script_hash: str,
) -> _Dict[str, object]:
    import math, numpy as np, cv2, torch
    from sam2_smoke import overlay_mask  # type: ignore
    from sam2_pilot import compute_iou, band_mask, nan_percentiles, nan_mean, write_per_frame_csv  # type: ignore
    clip_id = clip_cfg["id"]
    clip_dir = run_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    log_lines, log = base.create_logger()
    data_root: Path = context["data_root"]
    clip_path = data_root / "clips" / f"{clip_id}.mp4"
    gt_dir = data_root / "gt_frames" / clip_id
    if not clip_path.exists():
        raise SystemExit(f"Clip not found: {clip_path}")
    frames, (tgt_w, tgt_h), fps, seed_idx, seed_json, seed_box, seed_points, seed_labels, gt_boxes = base.load_frames_and_seed(
        clip_cfg, clip_path, gt_dir, int(clip_cfg.get("input_width", 1280))
    )
    device = context["device"]
    log(f"Processing {clip_id}")
    log(f"  Device: {device}")
    log(f"  Frames decoded: {len(frames)} @ {fps:.2f} fps -> {tgt_w}x{tgt_h}")
    # Detect shots
    try:
        shots_list = detect_shots(clip_path, total_frames=len(frames), fps=fps, method="adaptive", min_shot_len_s=0.5, adaptive_sensitivity=3)
        shot_bounds = [(int(s.start), int(s.end)) for s in shots_list]
    except Exception:
        shot_bounds = [(0, int(len(frames)))]
    # Build OWLv2 promptor
    args_local = _OWL_STATE.get("args")
    prompts = []
    if args_local and getattr(args_local, "prompts_file", None):
        try:
            prompts = [ln.strip() for ln in Path(args_local.prompts_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception:
            prompts = []
    if not prompts:
        prompts = PROMPT_TERMS
    dev = args_local.device if args_local and getattr(args_local, "device", None) in ("cuda", "cpu", "mps") else ("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    try:
        owl = OwlVitBoxPromptor(
            model_id=args_local.owlvit_model if args_local else "google/owlv2-base-patch16-ensemble",
            device=dev,
            prompts=prompts,
            score_thr=float(args_local.owlvit_score_thr) if args_local else 0.30,
        )
    except Exception:
        owl = None
    log(f"Autoprompt[OWL-ViT] model={getattr(args_local,'owlvit_model','N/A')} score_thr={getattr(args_local,'owlvit_score_thr','N/A')} device={dev}")
    log(f"Shots detected: {len(shot_bounds)}")
    total_frames = len(frames)
    masks_full: _List[np.ndarray] = [np.zeros((tgt_h, tgt_w), dtype=np.uint8) for _ in range(total_frames)]
    frame_runtimes: _Dict[int, float] = {}
    events: _List[_Dict[str, object]] = []
    fallback_mode = (args_local.autoprompt_fallback if args_local else "gt")
    # Per-shot sessions
    for i, (s, e) in enumerate(shot_bounds, start=1):
        if s >= e or s < 0 or e > total_frames:
            continue
        shot_frames = frames[s:e]
        pred = None
        preds2 = []
        if owl is not None and shot_frames:
            try:
                arr = np.asarray(shot_frames[0])
                if arr.ndim == 3 and arr.shape[2] == 3:
                    bgr0 = cv2.cvtColor(arr.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR)
                else:
                    bgr0 = None
            except Exception:
                bgr0 = None
            if bgr0 is not None:
                try:
                    preds2 = owl.predict_topk(bgr0, k=2) or []
                except Exception:
                    preds2 = []
        if preds2:
            pred = preds2[0]
        if pred is None:
            if fallback_mode == "skip":
                log(f"Shot {i}/{len(shot_bounds)}: owlvit NONE - skipping frames {s}-{e-1}")
                continue
            fb_box = None
            try:
                for fr in range(int(s), int(e)):
                    if fr in gt_boxes:
                        fb_box = gt_boxes[fr]
                        break
            except Exception:
                fb_box = None
            if fb_box is None:
                log(f"Shot {i}/{len(shot_bounds)}: owlvit NONE - no GT; skipping frames {s}-{e-1}")
                events.append({"shot_index": i, "start": s, "end": e, "mode": "fallback-none"})
                continue
            x0, y0, x1, y1 = map(int, fb_box)
            seed_boxes = [[[x0, y0, x1, y1]]]
            obj_id = 1
            session = processor.init_video_session(video=shot_frames, inference_device=device, processing_device=device, video_storage_device=device, dtype=context["dtype"])
            try:
                processor.add_inputs_to_inference_session(inference_session=session, frame_idx=0, obj_ids=obj_id, input_boxes=seed_boxes, input_points=None, input_labels=None)
            except Exception:
                pass
            log(f"Shot {i}/{len(shot_bounds)}: Fallback GT box: [{x0},{y0},{x1},{y1}] @ local_frame=0, obj_id={obj_id}")
            for off in range(0, e - s):
                mask, runtime_ms = base.infer_frame(model, processor, session, off, (tgt_w, tgt_h))
                masks_full[s + off] = mask
                frame_runtimes[s + off] = frame_runtimes.get(s + off, 0.0) + runtime_ms
            events.append({"shot_index": i, "start": s, "end": e, "mode": "fallback-gt", "box": [x0,y0,x1,y1]})
            continue
        x0, y0, x1, y1 = pred.as_int_tuple()
        x0, y0, x1, y1 = widen_banner_box(x0, y0, x1, y1, tgt_w, tgt_h)
        # Optional second face with vertical overlap and horizontal separation
        have_two = False
        if len(preds2) >= 2:
            x0b, y0b, x1b, y1b = preds2[1].as_int_tuple()
            x0b, y0b, x1b, y1b = widen_banner_box(x0b, y0b, x1b, y1b, tgt_w, tgt_h)
            h1 = max(1, y1 - y0); h2 = max(1, y1b - y0b)
            v_ov = max(0, min(y1, y1b) - max(y0, y0b)) / float(min(h1, h2))
            cx1 = 0.5 * (x0 + x1); cx2 = 0.5 * (x0b + x1b)
            sep_ok = abs(cx1 - cx2) >= 0.10 * tgt_w
            if v_ov >= 0.5 and sep_ok:
                have_two = True
        obj_id = 1
        x0, y0, x1, y1 = pred.as_int_tuple()
        x0, y0, x1, y1 = widen_banner_box(x0, y0, x1, y1, tgt_w, tgt_h)
        # Optional second face with vertical overlap and horizontal separation
        have_two = False
        if len(preds2) >= 2:
            x0b, y0b, x1b, y1b = preds2[1].as_int_tuple()
            x0b, y0b, x1b, y1b = widen_banner_box(x0b, y0b, x1b, y1b, tgt_w, tgt_h)
            h1 = max(1, y1 - y0); h2 = max(1, y1b - y0b)
            v_ov = max(0, min(y1, y1b) - max(y0, y0b)) / float(min(h1, h2))
            cx1 = 0.5 * (x0 + x1); cx2 = 0.5 * (x0b + x1b)
            sep_ok = abs(cx1 - cx2) >= 0.10 * tgt_w
            if v_ov >= 0.5 and sep_ok:
                have_two = True
        session = processor.init_video_session(video=shot_frames, inference_device=device, processing_device=device, video_storage_device=device, dtype=context["dtype"])
        
        try:
            # Seed a single object for stability
            obj_ids = 1
            input_boxes = [ [ [int(x0), int(y0), int(x1), int(y1)] ] ]
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=0,
                obj_ids=obj_ids,
                input_boxes=input_boxes,
                input_points=None,
                input_labels=None,
            )
            log(f"Shot {i}/{len(shot_bounds)}: OWL-ViT box: [{x0},{y0},{x1},{y1}] @ local_frame=0, obj_id=1")
        except Exception:
            pass
        nonempty_first = False
        for off in range(0, e - s):
            mask, runtime_ms = base.infer_frame(model, processor, session, off, (tgt_w, tgt_h))
            if off == 0:
                nonempty_first = bool((mask > 0).any())
            masks_full[s + off] = mask
            frame_runtimes[s + off] = frame_runtimes.get(s + off, 0.0) + runtime_ms



    frame_rows: _List[_Dict[str, object]] = []
    AREA_EPS = 1.0
    from sam2_smoke import load_labelme_polys  # type: ignore
    from sam2_pilot import polys_to_mask  # type: ignore
    label_paths = sorted(gt_dir.glob("frame_*.json"))
    for label_path in label_paths:
        frame_idx = int(label_path.stem.split("_")[-1])
        if frame_idx >= len(masks_full):
            continue
        src_w, src_h, polys = load_labelme_polys(label_path)
        if base.RUN_SPEC["metrics"]["scoring_policy"]["skip_frames_with_empty_gt"] and not polys:
            continue
        gt_mask = polys_to_mask(polys, src_w, src_h, tgt_w, tgt_h)
        if base.RUN_SPEC["metrics"]["scoring_policy"]["skip_frames_with_empty_gt"] and not gt_mask.any():
            continue
        pred_mask = masks_full[frame_idx] > 0
        gt_mask_bool = gt_mask > 0
        iou = compute_iou(pred_mask, gt_mask_bool)
        biou = compute_iou(band_mask(pred_mask), band_mask(gt_mask_bool))
        area = float(pred_mask.sum())
        frame_rows.append({"clip_id": clip_id, "frame_no": frame_idx, "iou": float(iou), "biou": float(biou), "area_px": float(area), "centroid_x": math.nan, "centroid_y": math.nan, "centroid_shift_px": math.nan, "centroid_shift_norm": math.nan, "is_empty": 1 if area <= AREA_EPS else 0, "runtime_ms": frame_runtimes.get(frame_idx, math.nan), "roi_w": tgt_w, "roi_h": tgt_h})
    # Overlay
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    overlay_path = clip_dir / ('overlay_%s_%s.mp4' % (clip_id, base.RUN_SPEC['run_id']))
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tgt_w, tgt_h))
    for frame, mask in zip(frames, masks_full):
        rgb = np.array(frame)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        bgr = overlay_mask(rgb, cleaned)
        writer.write(bgr)
    writer.release()
    per_frame_path = clip_dir / f"per_frame_{clip_id}_{base.RUN_SPEC['run_id']}.csv"
    write_per_frame_csv(per_frame_path, frame_rows, ["clip_id","frame_no","iou","biou","area_px","centroid_x","centroid_y","centroid_shift_px","centroid_shift_norm","is_empty","runtime_ms","roi_w","roi_h"])
    ious = [row["iou"] for row in frame_rows]
    biou_vals = [row["biou"] for row in frame_rows]
    jitter_vals = [val for val in (row["centroid_shift_norm"] for row in frame_rows) if isinstance(val, float) and not math.isnan(val)]
    area_vals = [row["area_px"] for row in frame_rows]
    empty_vals = [row["is_empty"] for row in frame_rows]
    iou_p25, iou_med, iou_p75 = nan_percentiles(ious, [25, 50, 75])
    biou_p25, biou_med, biou_p75 = nan_percentiles(biou_vals, [25, 50, 75])
    jitter_med, jitter_p95 = nan_percentiles(jitter_vals, [50, 95]) if jitter_vals else (math.nan, math.nan)
    area_mean = nan_mean(area_vals)
    import numpy as _np
    area_std = float(_np.std(_np.array(area_vals, dtype=_np.float32))) if not math.isnan(area_mean) else math.nan
    area_cv = math.nan if math.isnan(area_mean) or area_mean == 0 else (area_std / area_mean) * 100.0
    empty_pct = (sum(empty_vals) / len(empty_vals) * 100.0) if empty_vals else math.nan
    processed_count = len(frames)
    total_time_s = sum(frame_runtimes.get(idx, 0.0) for idx in range(len(frames))) / 1000.0
    fps_measured = processed_count / total_time_s if total_time_s > 0 else 0.0
    fps_theoretical = fps
    summary = {"clip_id": clip_id, "iou_median": iou_med, "iou_p25": iou_p25, "iou_p75": iou_p75, "biou_median": biou_med, "biou_p25": biou_p25, "biou_p75": biou_p75, "jitter_norm_median_pct": jitter_med, "jitter_norm_p95_pct": jitter_p95, "area_cv_pct": area_cv, "empty_frames_pct": empty_pct, "proc_fps_measured": fps_measured, "proc_fps_theoretical": fps_theoretical, "roi_w": tgt_w, "roi_h": tgt_h, "target_W": tgt_w, "target_H": tgt_h, "stride": int(clip_cfg.get("stride", 1))}
    # re_prompts rows
    re_prompt_path = clip_dir / f"re_prompts_{clip_id}_{base.RUN_SPEC['run_id']}.csv"
    try:
        rows=[]
        for ev in events:
            mode = ev.get("mode","owlvit")
            if mode=="owlvit":
                reason = f"owlvit ok label={ev.get('label','')} score={ev.get('score','')} box={ev.get('box',[])} first_nonempty={ev.get('first_nonempty','')}"
            elif mode=="fallback-gt":
                reason = f"fallback-gt box={ev.get('box',[])}"
            else:
                reason = str(mode)
            rows.append({"event_idx": int(ev.get("shot_index",0)), "frame_idx": int(ev.get("start",0)), "reasons": reason, "centroid_jump_px": "", "centroid_jump_pct": "", "area_change_pct": "", "empty_streak": ""})
        from sam2_pilot import write_reprompt_csv as _wr  # type: ignore
        _wr(re_prompt_path, rows)
    except Exception:
        pass
    # params + log
    params_path = clip_dir / f"params_{clip_id}_{base.RUN_SPEC['run_id']}.json"
    base.write_json(params_path, {"run_id": base.RUN_SPEC["run_id"], "clip_id": clip_id, "clip_path": str(clip_path), "gt_dir": str(gt_dir), "input_width": clip_cfg.get("input_width",1280), "stride": int(clip_cfg.get("stride",1)), "full_frame": clip_cfg.get("full_frame",False), "seed": {"frame_index": seed_idx, "json": str(seed_json), "bbox_pad_px": clip_cfg.get("seed",{}).get("bbox_pad_px",0), "box_xyxy": list(seed_box) if seed_box else [], "negatives": clip_cfg.get("seed",{}).get("negatives",{})}, "reseed": clip_cfg.get("reseed",{}), "model": {"weights": str(context["weights"]), "device": device, "dtype": str(context["dtype"])}, "script_hash": script_hash, "git_commit": git_commit})
    log_path = clip_dir / f"pilot_{clip_id}_{base.RUN_SPEC['run_id']}.log"
    with log_path.open('w') as f:
        for line in log_lines:
            f.write(f"{line}\n")
    return {"clip_id": clip_id, "summary": summary, "per_frame_path": per_frame_path, "per_frame_rows": len(frame_rows), "overlay_path": overlay_path, "re_prompts_path": re_prompt_path, "log_path": log_path, "params_path": params_path, "reseed_events": events, "fps_measured": fps_measured, "iou_median": iou_med, "biou_median": biou_med, "jitter_med": jitter_med, "empty_pct": empty_pct}




