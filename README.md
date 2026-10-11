# icrap-first-job

ICRAP 배치 잡 연습용 예제입니다. 나선형(spiral) 점 데이터를 직접 만들어 작은 MLP 로 두 갈래를 분류합니다.
`torch` 외에 다른 패키지나 외부 데이터가 필요 없습니다.

포털 배치 잡에서:

- 저장소: `https://github.com/loopin51/icrap-first-job`
- 명령: `python train.py --epochs 30` (옵션: `--lr`, `--hidden`)
- 이미지: 기본값 `PyTorch 2.14 · CUDA 13.0`, GPU 1장

결과물은 `$RUN_DIR` 에 `checkpoint.pt`, `result.json` 으로 남습니다.
선점되어 다시 시작하면 `checkpoint.pt` 에서 이어 갑니다.
