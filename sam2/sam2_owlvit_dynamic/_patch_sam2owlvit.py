from pathlib import Path
p=Path('sam2_owlvit.py')
s=p.read_text(encoding='utf-8')
start=s.find('def create_logger_wrapper():')
if start>=0:
    end=s.find('return lines, log', start)
    endline=s.find('\n', end)
    old=s[start:endline+1]
    new='def create_logger_wrapper():\n        lines, log = orig_create_logger()\n        _OWL_STATE["log"] = log\n        try:\n            if callable(log) and _OWL_STATE.get("args") and getattr(_OWL_STATE["args"], "auto_prompt", False):\n                log("Autoprompt[OWL-ViT] enabled: GT seeding disabled; using OWL-ViT only")\n        except Exception:\n            pass\n        return lines, log\n'
    s=s.replace(old,new)
s=s.replace('        proc.init_video_session = _init_video_session  # type: ignore[attr-defined]\n        proc.add_inputs_to_inference_session = _add_inputs  # type: ignore[attr-defined]\n        return proc\n','        from types import MethodType\n        proc.init_video_session = MethodType(_init_video_session, proc)  # type: ignore[attr-defined]\n        proc.add_inputs_to_inference_session = MethodType(_add_inputs, proc)  # type: ignore[attr-defined]\n        return proc\n')
Path('sam2_owlvit.py').write_text(s, encoding='utf-8')
print('OK')
