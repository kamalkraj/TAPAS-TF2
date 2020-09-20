# TAPAS-TF2



### Models TF2

Download [tapas_sqa_masklm_base_reset_tf2](https://drive.google.com/file/d/1uTVeUHEBXixjFe3IEV1Eb8gVue_VVBUc/view?usp=sharing)

## Convert weights
```bash
python converter.py --model_path=tapas_wtq_wikisql_sqa_masklm_medium_reset --do_reset --save_path=tapas_wtq_wikisql_sqa_masklm_medium_reset_tf2 --task=WTQ
```
```
  --[no]do_reset: Select model type for weight conversion.
    Reset refers to whether the parameter `reset_position_index_per_cell`was set to true or false during training.In general it's recommended to set it
    to true
    (default: 'true')
  --model_path: model_path for download models
  --save_path: save_path for saving converted weights
  --task: <SQA|WTQ|WIKISQL>: task for converison
    (default: 'SQA')
```