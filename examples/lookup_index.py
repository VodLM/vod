import datasets
import rich

from raffle_ds_research.tools import pipes
from raffle_ds_research.tools.pipes.utils.misc import iter_examples
from tests.tools.pipes.test_index_lookup import gen_corpus, gen_questions


def run():
    corpus = gen_corpus(seed=0)
    rich.print(f"> corpus_hash: {corpus._fingerprint}")
    questions = gen_questions(seed=0, corpus=corpus)
    rich.print(f"> questions_hash: {questions._fingerprint}")

    lookup_pipe = pipes.LookupIndexPipe(corpus=corpus, n_sections=10)
    rich.print(f"> lookup_pipe_hash: {datasets.fingerprint.Hasher.hash(lookup_pipe)}")

    # process a batch of questions using the pipe
    output = lookup_pipe(questions[:10])
    rich.print(output)

    # fetch the corresponding sections in the corpus
    for example in iter_examples(output):
        for result in iter_examples(example):
            pid = result[pipes.LookupIndexPipe._pid_key]
            # fetch the section content
            _ = corpus[pid]


if __name__ == "__main__":
    run()
