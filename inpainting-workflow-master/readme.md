


  ```
  /data/.venv/bin/python comfyui_run.py
  ```

```
python postprocess.py -i /data/fullworkflow/output -o /data/fullworkflow/postprocessed --top-mask /data/fullworkflow/sky_mask.png -j 1 --no-skip-existing  --seam-width 15 --pad 64

```

```
python comfyui_run.py   --workflow-json /data/fullworkflow/full-workflow-padded.json   --input-dir /data/fullworkflow/test/input/   --mask /data/fullworkflow/perspective_mask.png   --output-dir /data/fullworkflow/test/output   --workers 1   --server http://127.0.0.1:8188   --image-node-id 349   --mask-node-id 463 --overwrite
```

```
uv run  --with simple-lama-inpainting postprocess.py -i /data/fullwork
flow/test/output/ -o /data/fullworkflow/test/postprocessed --top-mask /data/fullworkflow/sky_mask_updated.png -j 
1 --no-skip-existing  --seam-width 20 --pad 64
```

```
python egoblur_infer.py   --input-dir /data/fullworkflow/test/postprocessed   --output-dir /data/fullworkflow/test/ego_results   --face-model models/egoblur_gen2/ego_blur_face_gen2.jit   --lp-model models/egoblur_gen2/ego_blur_lp_gen2.jit   --device auto   --workers 2   --blur soft   --blur-strength 0.90 --overwrite
```