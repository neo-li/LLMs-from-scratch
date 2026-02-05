# Chapter 2: Working with Text Data

## Main Chapter Code

- [ch02.ipynb](ch02.ipynb) contains all the code as it appears in the chapter

Key Contents

1) Build Word Dictionary, BytePair Enablding tiktoken.get_encoding("gpt2")
2) Build Torch Dataset, which return len, get_items
3) Build Dataloader define batch_size, shuffle, drop_last, num_worker
4) Create token_embedding_layer,  map Dataload inputs =>  token_embedding
5) Create position_embedding_layer, map Dataload max_length => position_embedding
6) input_embeddings = token_embeddings + position_embeddings. 

![Chapter 2 embedding visualization](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/19.webp)

## Optional Code

- [dataloader.ipynb](dataloader.ipynb) is a minimal notebook with the main data loading pipeline implemented in this chapter
