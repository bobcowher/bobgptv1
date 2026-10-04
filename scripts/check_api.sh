#!/usr/bin/env bash
#
# check_api.sh — curl checks for the OpenAI-compatible server (see docs/SERVING.md).
#
#   scripts/check_api.sh                     # against http://localhost:8000
#   BASE=http://lab:8000 scripts/check_api.sh
#
# Each check prints PASS/FAIL. Required fields come from openai-openapi
# (openapi.json, spec 2.3.0). Needs curl and jq.
set -uo pipefail

BASE="${BASE:-http://localhost:8000}"
MODEL="${MODEL:-bobgpt}"
pass=0; fail=0

check() {  # check "description" <jq filter that must be true> <json>; empty/invalid JSON fails
  if [ -n "$3" ] && printf '%s' "$3" | jq -e "$2" >/dev/null 2>&1; then
    echo "PASS  $1"; pass=$((pass + 1))
  else
    echo "FAIL  $1"; fail=$((fail + 1))
  fi
}

status_is() {  # status_is "description" <expected> <actual>
  if [ "$2" = "$3" ]; then echo "PASS  $1"; pass=$((pass + 1))
  else echo "FAIL  $1 (got HTTP $3, want $2)"; fail=$((fail + 1)); fi
}

post() {  # post <json body> -> writes body to $tmp, echoes HTTP status
  curl -s -o "$tmp" -w '%{http_code}' -X POST "$BASE/v1/chat/completions" \
       -H 'Content-Type: application/json' -d "$1"
}

tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT

if ! curl -s -o /dev/null "$BASE/v1/models"; then
  echo "Cannot connect to $BASE -- is the server running?"; exit 2
fi

echo "== GET /v1/models"
body=$(curl -s "$BASE/v1/models")
check "list wrapper: object=list, data is array" '.object == "list" and (.data | type == "array")' "$body"
check "model has id, object=model, integer created, owned_by" \
  '.data[0] | (.id | type == "string") and .object == "model" and (.created | type == "number" and floor == .) and (.owned_by | type == "string")' "$body"
check "model id is \"$MODEL\"" "any(.data[]; .id == \"$MODEL\")" "$body"

echo "== POST /v1/chat/completions (non-streaming)"
code=$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is a Python list?\"}],\"max_tokens\":200}")
body=$(cat "$tmp"); [ "$code" = 200 ] || body=""
status_is "HTTP 200" 200 "$code"
check "envelope: id, object=chat.completion, integer created, model" \
  '(.id | type == "string") and .object == "chat.completion" and (.created | type == "number" and floor == .) and .model == "'"$MODEL"'"' "$body"
check "choices[0]: index 0, message.role=assistant, string content" \
  '.choices[0] | .index == 0 and .message.role == "assistant" and (.message.content | type == "string")' "$body"
check "choices[0].message.refusal present (null)" '.choices[0].message | has("refusal") and .refusal == null' "$body"
check "choices[0].logprobs present (null)" '.choices[0] | has("logprobs") and .logprobs == null' "$body"
check "finish_reason is stop or length" '.choices[0].finish_reason | IN("stop", "length")' "$body"
check "usage: integer token counts, total = prompt + completion" \
  '.usage | (.prompt_tokens | type == "number") and (.completion_tokens | type == "number") and .total_tokens == .prompt_tokens + .completion_tokens' "$body"
check "completion_tokens <= max_tokens (200)" '.usage.completion_tokens <= 200' "$body"
check "content trimmed: no ### End / ### Question / ### Answer" '.choices[0].message.content | test("### (End|Question|Answer)") | not' "$body"
check "content has no leading/trailing whitespace" '.choices[0].message.content | . == (sub("^\\s+"; "") | sub("\\s+$"; ""))' "$body"
echo "      content end: $(jq -c '.choices[0].message.content[-100:]' "$tmp" 2>/dev/null)"
echo "      finish_reason: $(jq -r '.choices[0].finish_reason' "$tmp" 2>/dev/null), completion_tokens: $(jq -r '.usage.completion_tokens' "$tmp" 2>/dev/null)"

echo "== Request variants"
code=$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_completion_tokens\":5}")
status_is "max_completion_tokens accepted" 200 "$code"
[ "$code" = 200 ] && check "max_completion_tokens honored (<= 5)" '.usage.completion_tokens <= 5' "$(cat "$tmp")"

code=$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is a tuple?\"}]}],\"max_tokens\":10}")
status_is "content as array of text parts accepted" 200 "$code"

code=$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"system\",\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5}")
status_is "system message accepted" 200 "$code"

code=$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5,\"temperature\":0.2,\"top_p\":0.9,\"user\":\"x\",\"some_unknown_field\":true}")
status_is "extra/unknown fields ignored" 200 "$code"

code=$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count: one two three four\"}],\"max_tokens\":40,\"stop\":[\"the\",\"a\"]}")
[ "$code" = 200 ] && check "stop sequences not included in returned text" \
  '.choices[0].message.content | (contains("the") or contains(" a ")) | not' "$(cat "$tmp")"

echo "== Errors"
code=$(post "{\"model\":\"$MODEL\"}")
body=$(cat "$tmp")
status_is "missing messages -> HTTP 400 (FastAPI default is 422; needs a handler)" 400 "$code"
check "error shape: error.{message,type,param,code}" \
  '.error | (.message | type == "string") and (.type | type == "string") and has("param") and has("code")' "$body"

echo "== POST /v1/chat/completions (streaming)"
stream=$(curl -s -N -D "$tmp" -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is a dict?\"}],\"max_tokens\":20,\"stream\":true}")
check "Content-Type text/event-stream" '. == true' "$(grep -qi '^content-type: text/event-stream' "$tmp" && echo true || echo false)"
chunks=$(printf '%s\n' "$stream" | sed -n 's/^data: //p' | grep -v '^\[DONE\]$' | jq -s '.' 2>/dev/null || echo 'null')
check "every data: line (except [DONE]) is JSON" '. != null and length > 0' "$chunks"
check "last data: line is [DONE]" '. == true' "$([ "$(printf '%s\n' "$stream" | grep '^data: ' | tail -1)" = 'data: [DONE]' ] && echo true || echo false)"
check "chunks: object=chat.completion.chunk, same id and created on all" \
  'all(.[]; .object == "chat.completion.chunk") and (map(.id) | unique | length == 1) and (map(.created) | unique | length == 1)' "$chunks"
check "each choice has index, delta, finish_reason" 'all(.[].choices[]; has("index") and has("delta") and has("finish_reason"))' "$chunks"
check "first chunk delta.role = assistant" '.[0].choices[0].delta.role == "assistant"' "$chunks"
check "only the last choice chunk has finish_reason (stop|length)" \
  '[.[].choices[] | .finish_reason] | (.[:-1] | all(. == null)) and (.[-1] | IN("stop", "length"))' "$chunks"
echo "      text: $(printf '%s' "$chunks" | jq -r '[.[].choices[].delta.content // ""] | add' 2>/dev/null | cut -c1-120)"

stream=$(curl -s -N -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5,\"stream\":true,\"stream_options\":{\"include_usage\":true}}")
chunks=$(printf '%s\n' "$stream" | sed -n 's/^data: //p' | grep -v '^\[DONE\]$' | jq -s '.' 2>/dev/null || echo 'null')
check "include_usage: final chunk has choices=[] and usage" '.[-1] | (.choices == []) and (.usage.total_tokens | type == "number")' "$chunks"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
