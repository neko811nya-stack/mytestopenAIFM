# Minimal static file server for Windows that needs nothing installed (no Python / Node):
#   powershell -ExecutionPolicy Bypass -File serve.ps1      then open http://localhost:8000/
param([int]$Port = 8000)
$root = $PSScriptRoot
$mime = @{
  '.html' = 'text/html; charset=utf-8'; '.js' = 'text/javascript; charset=utf-8'; '.mjs' = 'text/javascript; charset=utf-8'
  '.css' = 'text/css; charset=utf-8'; '.json' = 'application/json'; '.jpg' = 'image/jpeg'; '.png' = 'image/png'
  '.svg' = 'image/svg+xml'; '.md' = 'text/plain; charset=utf-8'
}
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://localhost:$Port/")
try {
  $listener.Start()
} catch {
  Write-Host "Port $Port could not be opened (is another server already running?): $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}
Write-Host "Ocean is served at http://localhost:$Port/   (close this window or press Ctrl+C to stop)" -ForegroundColor Green
Start-Process "http://localhost:$Port/"
try {
  while ($listener.IsListening) {
    # wait asynchronously so Ctrl+C still works
    $task = $listener.GetContextAsync()
    while (-not $task.AsyncWaitHandle.WaitOne(200)) { }
    $ctx = $task.GetAwaiter().GetResult()
    $res = $ctx.Response
    try {
      $path = [Uri]::UnescapeDataString($ctx.Request.Url.AbsolutePath).TrimStart('/')
      if ($path -eq '' -or $path.EndsWith('/')) { $path += 'index.html' }
      $file = [IO.Path]::GetFullPath((Join-Path $root $path))
      if ($file.StartsWith($root, [StringComparison]::OrdinalIgnoreCase) -and (Test-Path -LiteralPath $file -PathType Leaf)) {
        $bytes = [IO.File]::ReadAllBytes($file)
        $ext = [IO.Path]::GetExtension($file).ToLower()
        if ($mime.ContainsKey($ext)) { $res.ContentType = $mime[$ext] } else { $res.ContentType = 'application/octet-stream' }
        # always serve the newest files (no stale cache after an update)
        $res.Headers.Add('Cache-Control', 'no-cache')
        $res.ContentLength64 = $bytes.Length
        $res.OutputStream.Write($bytes, 0, $bytes.Length)
      } else {
        $res.StatusCode = 404
      }
    } finally {
      $res.Close()
    }
  }
} finally {
  $listener.Stop()
}
