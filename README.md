# SSM Embeddings

This repository contains code to get embedding for entities annotated with terms from OWL ontologies based on the notion of semantic similarity.

The main idea is to compute semantic similarity between a large number of entity pairs (generated randomly from a set of entities) and then try to predict the similarity using a neural network. The input of for this network is the two entities that were compared, the output is compared with the actual similarity values, and the training phase tries to adjust the model parameters so that the neural network correctly (or as correctly as possible) estimates the various similarity values.

The repository was tested with a protein use case, where proteins are represented as the set of GO terms that annotate them.


## Semantic similarity (SSM)

Semantic similarity is computed from code inside the repository. Currently there are 14 distinct measures, and others may be implemented in the future.

Computing semantic similarity is by far the most expensive operation in terms of time. The computation of 200,000 semantic similarity vectors for protein pairs takes about 10h on a Intel Xeon Silver 4114 CPU (2.2GHz). It is planned to add multiprocessing capabilities to this process in the future, which should alleviate the problem.


## The neural network

The model that tries to learn semantic similarity consists of two parts:
- a concept set embedder with $H_e$ dense hidden layers, which converts a set of concepts into a final $N$-dimensional vector ($N$ and $H$ are parameterizable). All hidden layers are of the same shape.
- a mixer with $H_m$ hidden layers, which convert the concatenation of the embeddings of the two entities into a $O$-dimensional vector, where $O$ is the number of semantic similarity measures calculated in the training dataset

The model must be fed a pair of concept sets (see the section on one-hot encoding), which are both passed through the $H_e$ embedding layers. *Note* that there is only one set of embedding layers, and the weights used with each input are exactly the same.

The expectation is that the inner representations of the embedding layers is a useful representation of the entities, in a way that can be further explored with other machine learning endeavours. In this use case, we use the embeddings to predict whether two proteins interact.


## One-hot encoding

To feed a concept set into the model, it must be one-hot encoded. This procedure converts a set of concepts into a $C$-dimensional vector (where $C$ is the number of concepts in the ontology), where each value is either a 0 or a 1. Each dimension represents a concept, and a value of 0 means that the concept *is not* in the set, while a 1 means that the concept *is* in the set. Notice that we use the notion of subsumption in this schema: if a concept is in a set, we assume that so are all its ancestors. For example, in an ontology of anatomy, if a set contains the term `heart`, the values corresponding to the terms `heart`, `organ`, `body_part` etc. will all be 1.


## How to use

To use this code, you must follow the following steps:
1. Install the necessary python packages
1. Grab the necessary OWL ontology file and initial data
1. Generate the set of entities and their annotations
1. Save the ontology file as an SQLITE database
1. One-hot encode the entities in your dataset
1. Generate a random dataset of similarity values for pairs of entities
1. Train the embedder
1. Produce embeddings for all the entities

In what follows, we detail each step and give example code to reproduce our results. Each script contains several options and can be tuned for specific goals. We encourage you to explore the flags of each script (with `python <script> --help`) to see what options are available and how they can be used to tailor the code to your use case.


### Install the necessary python packages

This project heavily relies on some python packages that do the heavy lifting (OWL manipulation and machine-learning). As such, we need to first install a few requirements, as specified in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Notice that installing these packages may not be possible, if you have other versions of these packages in your system, or can even mess with your existing packages. A virtual environment (`venv`) or docker container approach is recommended, but advising on that is outside the scope of this project.


### Grab the necessary OWL ontology file and initial data

This and the next step are the one that will be specific to the use case. For our use case, we wanted to grab proteins with their GO annotations. As such, we downloaded:

- `data/uniprot-human-reviewed.txt.gz`: the set of reviewed human proteins from Uniprot;
- `data/9606.protein.info.v11.0.txt.gz`: the String DB file describing the human proteins in that database;
- `data/go-owl`: the GO OWL ontology.

For other use cases, other data will be required.

In the future, we might include a file `get_protein_data.sh` which performs these downloads automatically into the `data` directory.


### Generate the set of entities and their annotations

We include two files in the repository to read the data downloaded in the previous step and to extract the required annotation files.

An annotation file is a text file, which we mark with the extension `.annotations`, where each line contains two tab-separated fields:

1. the first field contains the name of an entity, which must be unique among all the entities in that file;
2. the second field contains a space-separated list of the IRIs of the concepts that annotate that entity.

An example of an annotation file:

```
OR4K3	http://purl.obolibrary.org/obo/GO_0016021 http://purl.obolibrary.org/obo/GO_0005886
O52A1	http://purl.obolibrary.org/obo/GO_0005887 http://purl.obolibrary.org/obo/GO_0005886
```

We generate these files (one for the Uniprot proteins and one for the String DB proteins) with two scripts specific to our use case

```bash
# Generate annotations for the uniprot proteins
python uniprot_go_annotations.py \
  data/uniprot-human-reviewed.txt.gz

# Generate annotations for the StringDB proteins
python string_db_annotations.py \
  data/9606.protein.info.v11.0.txt.gz \
  data/uniprot-human-reviewed.txt.gz
```


### Save the ontology file as an SQLITE database

All the steps from this one forward are entirely generic and can be applied verbatim (aside from filenames) to any use case.

Because the OWL ontology file is slow to parse, and it is used by multiple scripts, we start by converting it into an SQLITE triple store that can be digested by `owlready2`.

```bash
python to_sqlite.py data/go.owl
```


### One-hot encode the entities in your dataset

The next step is to one-hot encode all the annotated entities. Notice that one-hot encoding the entities is a three step procedure:

- first we need to generate a vocabulary file, which associates concepts with integer indices
- then we need to one-hot encode all the concepts in the ontology
- and only then do we have all the information to encode the entities.

One-hot encoding produces `.ohe` files, which are text files where each line contains the name of the entity, a tab character, and a space-separated list of integer indices that specify the concepts used to annotate the entities. Notice that, as discussed previously, we include the ancestor terms in the encoding.

```bash
# Generate the vocabulary
python vocabulary.py data/go.sqlite

# One-hot encode the concepts
python one_hot_encode.py classes data/go.sqlite data/go.vocab

# One hot-encode the entities
python one_hot_encode.py entities \
  data/go.classes.ohe \
  data/uniprot-human-reviewed.annotations

python one_hot_encode.py entities \
  data/go.classes.ohe \
  data/9606.protein.info.v11.0.annotations
```


### Generate a random dataset of similarity values for pairs of entities

Here we generate a dataset of entity pairs and their semantic similarity, as computed by several distinct measures. Note that currently we offer 14 measures of similarity, but this number may change in the future.

The larger the dataset, the more accurate the trained model will be in predicting semantic similarity values, so we recommend a large dataset. In our case, we generated 200,000 pairs of proteins with their semantic similarity values.

```bash
python dataset.py \
  data/go.sqlite \
  data/uniprot-human-reviewed.annotations \
  --size 200_000
```

This command generates the file `data/uniprot-human-reviewed.dataset`, which is a text file where each line is a tab-separated table. The first line contains a header with the name of the semantic similarity measures, and the subsequent lines contain the names of the entities being compared and the similarity values computed for them.

This step is by far the most time consuming, but it is also a one time process. Training on these values will be much (much!) quicker and can be performed multiple times with multiple different machine-learning parameters.

In the future, we might provide a quicker way to generate similarity, potentially with a compiled language (I'm currently looking at [`horned-owl`](https://github.com/phillord/horned-owl), a Rust crate, which should be much quicker than python's `owlready2`).


### Train the embedder

The central piece of the project is the learning phase, where we take the dataset generated in the previous step and, with a particularly crafted neural network (see the section above that explains this) learns to predict semantic similarity values between two entities.

```bash
python train.py \
  data/uniprot-human-reviewed.dataset \
  data/uniprot-human-reviewed.ohe
```

This command generates a new directory named after the time it began to execute (for example `outputs/20210525103959`). In this directory, there are three files:
- `config.json` contains the learning parameters of the experiment (such as the files for the dataset and one-hot-encoding), the parameters for the model architecture, the learning rate, the number of steps per epoch, etc.
- `events.out.tfevents.*` contains the tensorboard log file which stores the loss values (both train and eval) throughout the experiment
- `model-weights.pt` contains the weights learned by the model; these are the weights that were achieved with the lowest eval loss.

If you want to see the loss curve, you can either fire up a tensorboard session with the tfevents file above, or you can run our provided `loss_curve.py` file below, which produces the file `loss.png` in the same directory.

```bash
python loss_curve.py outputs/20210525103959
```


### Produce embeddings for all the entities

Once the model has been trained, it can be used to compute the embeddings for any number of entities (either the ones used in the training dataset or even others).

```bash
python embeddings.py \
  outputs/20210525103959 \
  data/9606.protein.info.v11.0.ohe
```

This produces several `embedding.X.tsv` files in the same directory, where `X` goes from $0$ to $H_e - 1$ and $H_e$ is the number of embedding layers of the model. These embeddings can now be used in other experiments as a way to represent the entities.

By default, this script normalizes the embeddings in such a way that each dimension has mean 0 and standard deviation 1. This should not, in principle, change the power of the embeddings, while it facilitates the other experiments that will spawn from these results.

The format of these files is TSV (tab-separated values), where each line represents one entity, and contains the name of the entity and the $N$ values of the embedding for that entity.
