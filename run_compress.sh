#python main.py models="lz-compress-plus-3","lz-compress-plus-5","lz-compress-plus-10" rounds=3 epochs=3

#python main.py models="lz-vanilla-compress-3","lz-vanilla-compress-5","lz-vanilla-compress-10" rounds=3 epochs=3

#python main.py models="lz-compress-plus-mean-3","lz-compress-plus-mean-5","lz-compress-plus-mean-10" rounds=3 epochs=3

# Task to be run
#python main.py models="lz-compress-pre-plus-3","lz-compress-pre-plus-5","lz-compress-pre-plus-10" rounds=3 epochs=5

python main.py models="self-lz-compress-pre-train-3","self-lz-compress-plus-3","self-lz-compress-pre-plus-3" rounds=3 epochs=3
