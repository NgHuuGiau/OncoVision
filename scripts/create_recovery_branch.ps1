$token = $env:GITHUB_TOKEN
if (-not $token) {
    Write-Error "GITHUB_TOKEN not set"
    exit 1
}

$headers = @{
    'Accept'        = 'application/vnd.github.v3+json'
    'Authorization' = "token $token"
}

$body = @{
    ref = 'refs/heads/recover-old-history'
    sha = '2f0562f1e7cb188b46720e244979dd767d16c7f1'
} | ConvertTo-Json -Depth 3

$response = Invoke-WebRequest -Uri 'https://api.github.com/repos/NgHuuGiau/OncoVision/git/refs' -Method POST -Headers $headers -Body $body -ContentType 'application/json' -UseBasicParsing
Write-Host "Status: $($response.StatusCode)"
Write-Host $response.Content