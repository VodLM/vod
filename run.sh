poetry run train +recipe=frank-bert-base builder.name=frank.A.en.pos exp_suffix="-pos"
poetry run train +recipe=frank-t5-base builder.name=frank.A.en.pos exp_suffix="-pos"
poetry run train +recipe=frank-t5-large builder.name=frank.A.en.pos exp_suffix="-pos"
poetry run train +recipe=frank-bert-base builder.name=frank.A.en exp_suffix="-full"
poetry run train +recipe=frank-t5-base builder.name=frank.A.en exp_suffix="-full"
poetry run train +recipe=frank-t5-large builder.name=frank.A.en exp_suffix="-full"
