python -m train.train_condmdi --keyframe_conditioned \
 --log_interval 5 \
 --save_interval 1000 \
 --eval_during_training \
 --eval_split test \
 --keyframe_guidance_param 1.0 \
 --batch_size 64
