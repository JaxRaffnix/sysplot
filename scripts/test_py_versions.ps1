$versions = uv run python scripts/python_matrix.py | ConvertFrom-Json

foreach ($v in $versions) {
    Write-Host "`n== Python $v =="

    uv python install $v | Out-Null

    uv sync --all-extras --dev

    uv run --python $v pytest
    if ($LASTEXITCODE -ne 0) {
        Write-Error "FAILED on Python $v"
        exit 1
    }
}

Write-Host "`nAll Python versions passed."