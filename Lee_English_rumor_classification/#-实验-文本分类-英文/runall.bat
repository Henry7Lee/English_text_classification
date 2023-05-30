call activate torch101
for %%a in (bert_CNN bert_DPCNN bert_RCNN bert_RNN) do (
python run.py --model %%a
)
pause
