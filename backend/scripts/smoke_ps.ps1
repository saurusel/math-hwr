# encoding: utf-8
param(
  [string]$RunId = "",
  [string]$Ckpt = "checkpoint_latest.pt",
  [int]$N = 50,
  [ValidateSet('greedy','beam')][string]$DecodeType = 'greedy',
  [int]$BeamWidth = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Force UTF-8 console for Windows PowerShell 5.x
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding  = [System.Text.UTF8Encoding]::new($false)
$PSDefaultParameterValues['*:Encoding'] = 'utf8'
chcp 65001 | Out-Null

# Fix mojibake in strings (Ã·, Ã×, stray Â, etc.)
function Fix-Text([string]$s) {
  if ($null -eq $s) { return "" }

  $divSym   = [string][char]0x00F7   # ÷
  $mulSym   = [string][char]0x00D7   # ×
  $mojiDiv  = [string]([char]0x00C3) + [char]0x00B7   # "Ã·"
  $mojiMul1 = [string]([char]0x00C3) + [char]0x0097   # C3 97 path
  $mojiMul2 = [string]([char]0x00C3) + [char]0x2014   # "Ã—" seen as C3 + em-dash
  $s = $s -replace [regex]::Escape($mojiDiv),  $divSym
  $s = $s -replace [regex]::Escape($mojiMul1), $mulSym
  $s = $s -replace [regex]::Escape($mojiMul2), $mulSym

  # remove stray "Â" (C2) characters
  $s = $s -replace [string][char]0x00C2, ""

  # If still contains "Ã", try Latin1->UTF8 re-interpretation
  if ($s -like "*$([string][char]0x00C3)*") {
    try {
      $bytes = [Text.Encoding]::GetEncoding(28591).GetBytes($s) # Latin-1
      $s2    = [Text.Encoding]::UTF8.GetString($bytes)
      if ($s2 -notlike "*$([string][char]0x00C3)*") { $s = $s2 }
    } catch { }
  }
  return $s
}

# Levenshtein (chars) O(m) memory
function Get-Levenshtein([string]$a, [string]$b) {
  if ($null -eq $a) { $a = "" }
  if ($null -eq $b) { $b = "" }
  $a = Fix-Text $a; $b = Fix-Text $b
  [int]$n = $a.Length; [int]$m = $b.Length
  $prev = New-Object int[] ($m + 1)
  $curr = New-Object int[] ($m + 1)
  for ($j=0; $j -le $m; $j++) { $prev[$j] = $j }
  for ($i=1; $i -le $n; $i++) {
    $curr[0] = $i
    for ($j=1; $j -le $m; $j++) {
      $cost = [int]([char]$a[$i-1] -ne [char]$b[$j-1])
      $del = $prev[$j] + 1
      $ins = $curr[$j-1] + 1
      $sub = $prev[$j-1] + $cost
      $curr[$j] = [Math]::Min($del, [Math]::Min($ins, $sub))
    }
    $tmp = $prev; $prev = $curr; $curr = $tmp
  }
  return $prev[$m]
}

# Levenshtein (tokens) O(m) memory
function Get-LevTokens([string[]]$A, [string[]]$B) {
  if ($null -eq $A) { $A = @() }
  if ($null -eq $B) { $B = @() }
  $A = $A | ForEach-Object { Fix-Text $_ }
  $B = $B | ForEach-Object { Fix-Text $_ }
  [int]$n = $A.Count; [int]$m = $B.Count
  $prev = New-Object int[] ($m + 1)
  $curr = New-Object int[] ($m + 1)
  for ($j=0; $j -le $m; $j++) { $prev[$j] = $j }
  for ($i=1; $i -le $n; $i++) {
    $curr[0] = $i
    for ($j=1; $j -le $m; $j++) {
      $cost = [int]([string]$A[$i-1] -ne [string]$B[$j-1])
      $del = $prev[$j] + 1
      $ins = $curr[$j-1] + 1
      $sub = $prev[$j-1] + $cost
      $curr[$j] = [Math]::Min($del, [Math]::Min($ins, $sub))
    }
    $tmp = $prev; $prev = $curr; $curr = $tmp
  }
  return $prev[$m]
}

# Load first N labels as UTF-8
$labels = Get-Content -Encoding UTF8 data/synth/val/labels.jsonl -TotalCount $N |
          ForEach-Object { $_ | ConvertFrom-Json }

$exact=0; $char_err=0; $char_tot=0; $tok_err=0; $tok_tot=0
$mismatches = @(); $i=0

foreach($row in $labels){
  $id     = $row.id
  $target = Fix-Text $row.target

  $decode = if ($DecodeType -eq 'beam') { @{ type='beam'; beam_width=$BeamWidth } }
            else                        { @{ type='greedy' } }

  $body = @{
    run_id    = $RunId
    checkpoint= $Ckpt
    data      = @{ img_h=64; img_w_max=512; invert=$true; pad_mode="right" }
    decode    = $decode
    image     = @{ path = "data/synth/val/images/$id.png" }
  } | ConvertTo-Json -Depth 6

  try {
    $res  = Invoke-RestMethod -Method Post `
            -Uri http://127.0.0.1:8000/api/predict2/run `
            -Body $body -ContentType "application/json; charset=utf-8" `
            -Headers @{ "Accept"="application/json"; "Accept-Charset"="utf-8" }

    $pred = Fix-Text $res.text

    $i++
    Write-Host ("[{0}/{1}] {2} -> " -f $i,$labels.Count,$id) -NoNewline
    Write-Host $pred

    if($pred -eq $target){ $exact++ }
    $char_err += Get-Levenshtein $pred $target
    $char_tot += $target.Length
    $tok_err  += Get-LevTokens ($pred -split ' ') ($target -split ' ')
    $tok_tot  += ($target -split ' ').Count

    if($pred -ne $target){
      $mismatches += [pscustomobject]@{ id=$id; target=$target; pred=$pred }
    }
  } catch {
    $i++
    Write-Host ("[{0}/{1}] {2} -> <ERROR: {3}>" -f $i, $labels.Count, $id, $_.Exception.Message)
  }
}

# --- сводка и сохранение ---
$cer        = if($char_tot -gt 0){ [Math]::Round($char_err / $char_tot, 4) } else { 0 }
$wer        = if($tok_tot  -gt 0){ [Math]::Round($tok_err  / $tok_tot,  4) } else { 0 }
$exact_rate = if($labels.Count -gt 0){ [Math]::Round($exact / $labels.Count, 4) } else { 0 }

# безопасно берём первые 10 (если 0 — вернётся пусто)
$mm = $mismatches | Select-Object -First 10

$report = [pscustomobject]@{
  run_id     = $RunId
  ckpt       = $Ckpt
  n          = $labels.Count
  decode     = $DecodeType
  beam_width = $BeamWidth
  exact      = $exact_rate
  cer        = $cer
  wer        = $wer
  mismatches = $mm
}

$dir = "runs/$RunId/eval"; if (!(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
$fname = if ($DecodeType -eq 'beam') { "predict2_smoke_beam.json" } else { "predict2_smoke_greedy.json" }
$report | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 "$dir/$fname"

Write-Host ("SMOKE[{0}]: exact={1}  cer={2}  wer={3}  (n={4})" -f $DecodeType, $exact_rate, $cer, $wer, $labels.Count)
