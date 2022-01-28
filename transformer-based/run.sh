python word_language.py --data /home/shenghao/dataset/wikitext-2 \
        --model Transformer \
	--emsize 512\
  	--nhid 1024 \
	--nlayers 24 \
	--lr 0.2 \
	--clip 0.25 \
	--epochs 40 \
	--batch_size 4 \
	--bptt 128 \
       	--nhead 16 \
	--cuda