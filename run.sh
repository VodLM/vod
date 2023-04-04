poetry run train +recipe=frank-bert-base
poetry run train +recipe=frank-t5-base
poetry run train +recipe=frank-t5-large
poetry run train +recipe=frank-bert-base builder.include_only_positive_sections=false exp_suffix="-full"
poetry run train +recipe=frank-t5-base builder.include_only_positive_sections=false exp_suffix="-full"
poetry run train +recipe=frank-t5-large builder.include_only_positive_sections=false exp_suffix="-full"
