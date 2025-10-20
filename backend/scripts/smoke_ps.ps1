param(
  [string]$RunId      = "20251020_102849_crnn_ctc",
  [string]$Ckpt       = "checkpoint_latest.pt",
  [int]   $N          = 50,
  [ValidateSet("greedy","beam")]
  [string]$DecodeType = "greedy",
  [int]   $BeamWidth  = 5
)

$ErrorActionPreference = 'Stop'

# --- UTF-8 консоль и дефолтные кодировки ---
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding  = [System.Text.UTF8Encoding]::new($false)
$PSDefaultParameterValues['*:Encoding'] = 'utf8'
chcp 65001 | Out-Null

# --- Levenshtein (симв.) ---
function Get-Levenshtein([string]$a, [string]$b) {
  if ($null -eq $a) { $a = "" }
  if ($null -eq $b) { $b = "" }
  [int]$n = $a.Length; [int]$m = $b.Length
  $prev = New-Object int[] ($m + 1)
  $curr = New-Object int[] ($m + 1)
  for ($j=0; $j -le $m; $j++) { $prev[$j] = $j }
  for ($i=1; $i -le $n; $i++) {
    $curr[0] = $i
    for ($j=1; $j -le $m; $j++) {
      $cost = [int]($a[$i-1] -ne $b[$j-1])
      $del = $prev[$j] + 1
      $ins = $curr[$j-1] + 1
      $sub = $prev[$j-1] + $cost
      $curr[$j] = [Math]::Min($del, [Math]::Min($ins, $sub))
    }
    $tmp = $prev; $prev = $curr; $curr = $tmp
  }
  return $prev[$m]
}

# --- Levenshtein (по токенам) ---
function Get-LevTokens([string[]]$A, [string[]]$B) {
  if ($null -eq $A) { $A = @() }
  if ($null -eq $B) { $B = @() }
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

# --- читаем первые N меток как UTF-8 ---
$labels = Get-Content -Encoding UTF8 data/synth/val/labels.jsonl -TotalCount $N |
          ForEach-Object { $_ | ConvertFrom-Json }

# --- прогон + метрики ---
$exact=0; $char_err=0; $char_tot=0; $tok_err=0; $tok_tot=0
$mismatches = @(); $i = 0

foreach($row in $labels){
  $id     = $row.id
  $target = $row.target

  $decodeCfg = if ($DecodeType -eq "beam") { @{ type="beam"; beam_width=$BeamWidth } }
               else { @{ type="greedy" } }

  $body = @{
    run_id    = $RunId
    checkpoint= $Ckpt
    data      = @{ img_h=64; img_w_max=512; invert=$true; pad_mode="right" }
    decode    = $decodeCfg
    image     = @{ path = "data/synth/val/images/$id.png" }
  } | ConvertTo-Json -Depth 6

  try {
    $res  = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/predict2/run -Body $body -ContentType "application/json"
    $pred = $res.text

    # только для печати — нормализуем возможный ANSI-«мусор» в ÷
    $pred_disp = $pred -replace 'Ã·','÷'

    $i++; Write-Host ("[{0}/{1}] {2} -> {3}" -f $i,$labels.Count,$id,$pred_disp)

    if($pred -eq $target){ $exact++ }
    $char_err += Get-Levenshtein $pred $target
    $char_tot += $target.Length
    $tok_err  += Get-LevTokens ($pred -split ' ') ($target -split ' ')
    $tok_tot  += ($target -split ' ').Count
    if($pred -ne $target){ $mismatches += [pscustomobject]@{ id=$id; target=$target; pred=$pred } }
  }
  catch {
    $i++; Write-Host ("[{0}/{1}] {2} -> <ERROR: {3}>" -f $i,$labels.Count,$id,$_.Exception.Message)
  }
}

$cer        = if($char_tot -gt 0){ [Math]::Round($char_err / $char_tot, 4) } else { 0 }
$wer        = if($tok_tot  -gt 0){ [Math]::Round($tok_err  / $tok_tot,  4) } else { 0 }
$exact_rate = [Math]::Round($exact / $labels.Count, 4)

Write-Host ("SMOKE[{0}]: exact={1}  cer={2}  wer={3}  (n={4})" -f $DecodeType,$exact_rate,$cer,$wer,$labels.Count)

# --- сохраняем отчёт ---
$report = [pscustomobject]@{
  run_id=$RunId; ckpt=$Ckpt; n=$labels.Count
  exact=$exact_rate; cer=$cer; wer=$wer
  mismatches=$mismatches[0..([Math]::Min(9,[Math]::Max(0,$mismatches.Count-1)))]
}
$dir="runs/$RunId/eval"; if(!(Test-Path $dir)){ New-Item -ItemType Directory -Path $dir | Out-Null }
$report | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 "$dir/predict2_smoke_$DecodeType.json"
