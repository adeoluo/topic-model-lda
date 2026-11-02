# Records docs/demo/http-topics.gif from a real run: `vhs docs/demo/http-topics.tape` from the repo root.
Output docs/demo/http-topics.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 480
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate` Enter
Sleep 2s
Type `python -m topic_model_lda serve --port 18330 >/dev/null 2>&1 &` Enter
Sleep 2s
Sleep 8.0s
Type `clear` Enter
Sleep 1s
Show
Type `curl -s localhost:18330/topics | jq -c` Enter
Sleep 3s
Type `curl -s localhost:18330/evaluate | jq -c` Enter
Sleep 2.5s
Type `curl -s localhost:18330/metrics | head -n 6` Enter
Sleep 3s
Hide
Type `kill %1` Enter
Sleep 1s
