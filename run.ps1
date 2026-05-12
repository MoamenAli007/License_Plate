# Smart Campus - License Plate Detector
# Run this script instead of calling detect_rec.py directly.
# It sets UTF-8 so Arabic text displays correctly.

# Enable UTF-8 output
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

# Pass all arguments through to detect_rec.py
$args_str = $args -join ' '

& .\venv312\Scripts\python detect_rec.py --weights best.pt $args
