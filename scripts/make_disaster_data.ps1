# 为 reset 验证造测试数据
Write-Host "Preparing disaster test data..."

New-Item -ItemType Directory -Force -Path "data/raw/precious_dataset/images" | Out-Null
New-Item -ItemType Directory -Force -Path "data/raw/precious_dataset/labels" | Out-Null
1..200 | ForEach-Object {
    "fake image $_" | Set-Content "data/raw/precious_dataset/images/img_$_.jpg"
    "0 0.5 0.5 0.3 0.4" | Set-Content "data/raw/precious_dataset/labels/img_$_.txt"
}
Write-Host "  [OK] data/raw/precious_dataset/ - 400 files"

New-Item -ItemType Directory -Force -Path "data/processed/demo/train" | Out-Null
New-Item -ItemType Directory -Force -Path "data/processed/demo/val" | Out-Null
New-Item -ItemType Directory -Force -Path "data/processed/demo/test" | Out-Null
1..120 | ForEach-Object { "img $_" | Set-Content "data/processed/demo/train/img_$_.txt" }
1..20  | ForEach-Object { "img $_" | Set-Content "data/processed/demo/val/img_$_.txt" }
1..20  | ForEach-Object { "img $_" | Set-Content "data/processed/demo/test/img_$_.txt" }
Write-Host "  [OK] data/processed/demo/ - 160 files"

New-Item -ItemType Directory -Force -Path "runs/exp_2026_05_10" | Out-Null
fsutil file createnew "runs/exp_2026_05_10/best.pt" 2147483648 | Out-Null
fsutil sparse setflag "runs/exp_2026_05_10/best.pt" | Out-Null
Write-Host "  [OK] runs/exp_2026_05_10/best.pt - 2 GB sparse"

New-Item -ItemType Directory -Force -Path "runs/exp_2026_05_10/tb_logs" | Out-Null
1..5000 | ForEach-Object { "step $_ loss" | Set-Content "runs/exp_2026_05_10/tb_logs/event.$_" }
Write-Host "  [OK] runs/exp_2026_05_10/tb_logs/ - 5000 files"

New-Item -ItemType Directory -Force -Path "models/trained" | Out-Null
1..3 | ForEach-Object { "fake weights $_" | Set-Content "models/trained/model_v$_.pt" }
Write-Host "  [OK] models/trained/ - 3 weights"

New-Item -ItemType Directory -Force -Path "apps/platform/logging/training/2026-05-10" | Out-Null
1..50 | ForEach-Object { "training run $_" | Set-Content "apps/platform/logging/training/2026-05-10/run-$_.log" }
Write-Host "  [OK] apps/platform/logging/ - 50 logs"

Write-Host "Done."