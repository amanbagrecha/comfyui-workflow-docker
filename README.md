# ComfyUI Inpainting and Privacy Blur Pipeline

This repo runs the host-native multi-GPU 360 panorama pipeline with `./run_multi_gpu_pipeline.sh`; on a normal prepared machine, set `SRC` and `FINAL_OUTPUT_DIR` and run it from the repo root. The runtime lives in `.venv`, `ComfyUI`, `models`, `native_data`, and `logs`, and `run_multi_gpu_pipeline.sh` still supports host bootstrap unless you explicitly skip it.

For rented or snapshotted Vast machines, clone `main`, run `bash scripts/bootstrap.sh`, and pass `AWS_DOWNLOAD_PROFILE`, `AWS_UPLOAD_PROFILE`, `DOWNLOAD_S3_URI`, `DOWNLOAD_DEST_DIR`, `SRC`, and `FINAL_OUTPUT_DIR` as needed; the bootstrap script installs `aws` and `opencode` only if missing, optionally downloads data, and then runs `./run_multi_gpu_pipeline.sh` with `SKIP_HOST_BOOTSTRAP=1` by default so a pre-baked machine can go straight to execution.

For fast S3 downloads, use `inpainting-workflow-master/s3_parallel_download.py`:
```bash
python inpainting-workflow-master/s3_parallel_download.py --bucket BUCKET --prefix PREFIX --dest DEST
```
