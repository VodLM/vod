import datasets
import rich

from raffle_ds_research.tools import pipes
from raffle_ds_research.tools.pipes.utils.misc import iter_examples
from tests.tools.pipes.test_index_lookup import gen_corpus, gen_questions


def run():
    keys = ["kb_id", "answer_id", "section_id"]
    corpus = gen_corpus(size_per_step=3, keys=keys)
    rich.print(f"> corpus_hash: {corpus._fingerprint}")
    rich.print(corpus)
    questions = gen_questions(seed=1, corpus=corpus, num_questions=10, last_level_link_prob=0.5)
    rich.print(f"> questions_hash: {questions._fingerprint}")
    rich.print(questions)

    lookup_pipe = pipes.LookupIndexPipe(corpus=corpus, keys=keys)
    rich.print(f"> lookup_pipe_hash: {datasets.fingerprint.Hasher.hash(lookup_pipe)}")

    # process a batch of questions using the pipe
    batch = questions[:3]
    rich.print("=== INPUT ===")
    rich.print(batch)
    output = lookup_pipe(batch)
    rich.print("=== OUTPUT ===")
    rich.print(output)

    # fetch the corresponding sections in the corpus
    for example in iter_examples(output):
        for result in iter_examples(example):
            pid = result[lookup_pipe._output_idx_name]
            # fetch the section content
            row = corpus[int(pid)]
            rich.print(row)
            break
        break


if __name__ == "__main__":
    run()
