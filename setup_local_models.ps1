# Setup Local Models for Docling
# This script organizes local models to avoid HuggingFace API calls

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelsDir = Join-Path $scriptDir "models"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Docling Local Models Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create models directory
Write-Host "[1/3] Creating models directory..." -ForegroundColor Yellow
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
    Write-Host "  Created: $modelsDir" -ForegroundColor Green
} else {
    Write-Host "  Already exists: $modelsDir" -ForegroundColor Green
}

# Step 2: Move/rename docling-layout-heron folder
Write-Host ""
Write-Host "[2/3] Setting up Layout Model (HERON)..." -ForegroundColor Yellow
$sourceHeron = Join-Path $scriptDir "docling-layout-heron"
$targetHeron = Join-Path $modelsDir "ds4sd--docling-layout-heron"

if (Test-Path $sourceHeron) {
    if (Test-Path $targetHeron) {
        Write-Host "  Layout model already exists" -ForegroundColor Green
    } else {
        Move-Item -Path $sourceHeron -Destination $targetHeron
        Write-Host "  Moved layout model successfully" -ForegroundColor Green
    }
} elseif (Test-Path $targetHeron) {
    Write-Host "  Layout model already exists" -ForegroundColor Green
} else {
    Write-Host "  Layout model not found" -ForegroundColor Red
}

# Step 3: Check for TableFormer model
Write-Host ""
Write-Host "[3/3] Checking Table Structure Model (TableFormer)..." -ForegroundColor Yellow
$targetTableFormer = Join-Path $modelsDir "ds4sd--docling-models"

if (Test-Path $targetTableFormer) {
    Write-Host "  TableFormer model exists" -ForegroundColor Green
} else {
    Write-Host "  TableFormer model not found" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  To download TableFormer, run ONE of these:" -ForegroundColor Yellow
    Write-Host "    python -m docling.cli.models download tableformer -o ""$modelsDir""" -ForegroundColor White
    Write-Host "  OR" -ForegroundColor Yellow
    Write-Host "    huggingface-cli download ds4sd/docling-models --local-dir ""$targetTableFormer""" -ForegroundColor White
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Models directory: $modelsDir" -ForegroundColor White
Write-Host ""
Write-Host "Layout Model:" -ForegroundColor White
if (Test-Path $targetHeron) {
    Write-Host "  [OK] $targetHeron" -ForegroundColor Green
} else {
    Write-Host "  [MISSING] $targetHeron" -ForegroundColor Red
}

Write-Host ""
Write-Host "TableFormer Model:" -ForegroundColor White
if (Test-Path $targetTableFormer) {
    Write-Host "  [OK] $targetTableFormer" -ForegroundColor Green
} else {
    Write-Host "  [MISSING] $targetTableFormer" -ForegroundColor Red
}

Write-Host ""
if ((Test-Path $targetHeron) -and (Test-Path $targetTableFormer)) {
    Write-Host "All models ready! Docling will use local models." -ForegroundColor Green
} else {
    Write-Host "Some models missing. Download them manually." -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
