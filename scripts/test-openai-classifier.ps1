# Test Qwen-based issue classifier locally via Alibaba Cloud ModelStudio (Bailian)
# 1. Fill in your API key below
# 2. Run: powershell -ExecutionPolicy Bypass -File scripts/test-openai-classifier.ps1

# ====== FILL IN YOUR KEY HERE ======
$API_KEY = "sk-d087c2e875754044af5214d0dbaf3387"
# ===================================

# Alibaba Cloud ModelStudio (Bailian) - International/Singapore region, OpenAI-compatible
$API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
$MODEL   = "qwen-plus"

# Sample issue (change these to test different cases)
$ISSUE_TITLE = "Suggestion: add batch trace upload support in UI"
$ISSUE_BODY = @"
Currently users can only add one trace at a time in the trace analysis box.
It would be helpful to support batch uploading multiple traces at once,
either via drag-and-drop or a file picker that allows multi-select.

This would save time when analyzing multiple inference runs.
"@

# Build the prompt (same as the workflow)
$prompt = @"
You are a GitHub issue classifier for Hyperloom, an AI-powered performance optimization platform for AMD GPUs.

Analyze this issue and return ONLY a JSON object (no markdown fences):
- "type": one of "type:bug", "type:feature", "type:question", "type:task", "type:docs"
- "domains": array from ["domain:inference", "domain:training", "domain:mcp", "domain:ui"], can be empty

Issue title: $ISSUE_TITLE

Issue body:
$ISSUE_BODY
"@

# Build request
$headers = @{
    "Authorization" = "Bearer $API_KEY"
    "Content-Type"  = "application/json"
}

$body = @{
    model       = $MODEL
    messages    = @(@{role = "user"; content = $prompt})
    temperature = 0
    max_tokens  = 150
} | ConvertTo-Json -Depth 5

Write-Host "===== Sending request =====" -ForegroundColor Cyan
Write-Host "Endpoint: $API_URL"
Write-Host "Model: $MODEL"
Write-Host "Issue title: $ISSUE_TITLE"
Write-Host ""

try {
    $response = Invoke-WebRequest `
        -Uri $API_URL `
        -Method POST `
        -Headers $headers `
        -Body $body `
        -UseBasicParsing

    $json = $response.Content | ConvertFrom-Json
    $content = $json.choices[0].message.content

    Write-Host "===== HTTP Status: $($response.StatusCode) =====" -ForegroundColor Green
    Write-Host ""
    Write-Host "===== LLM raw output =====" -ForegroundColor Cyan
    Write-Host $content
    Write-Host ""

    # Parse the JSON output (strip markdown fences if present)
    $cleaned = $content -replace '^```(json)?', '' -replace '```$', ''
    $cleaned = $cleaned.Trim()

    try {
        $parsed = $cleaned | ConvertFrom-Json
        Write-Host "===== Parsed labels =====" -ForegroundColor Green
        Write-Host "type:    $($parsed.type)"
        Write-Host "domains: $($parsed.domains -join ', ')"
    } catch {
        Write-Host "Could not parse LLM output as JSON" -ForegroundColor Yellow
        Write-Host "Cleaned: $cleaned" -ForegroundColor Yellow
    }

    if ($json.usage) {
        Write-Host ""
        Write-Host "===== Token usage =====" -ForegroundColor Cyan
        Write-Host "Prompt tokens:     $($json.usage.prompt_tokens)"
        Write-Host "Completion tokens: $($json.usage.completion_tokens)"
        Write-Host "Total tokens:      $($json.usage.total_tokens)"
    }

} catch {
    Write-Host "===== ERROR =====" -ForegroundColor Red
    Write-Host "Status: $($_.Exception.Response.StatusCode.value__)"
    Write-Host "Body:"
    Write-Host $_.ErrorDetails.Message
}
