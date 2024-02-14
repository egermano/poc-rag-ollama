import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { Ollama } from '@langchain/community/llms/ollama';
import { Chroma } from '@langchain/community/vectorstores/chroma';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts';
import { pull } from 'langchain/hub';
// import { YoutubeLoader } from 'langchain/document_loaders/web/youtube';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';

const model = new Ollama({
  baseUrl: 'http://localhost:11434', // Default value
  model: 'llama2', // Default value
});

const embeddings = new OllamaEmbeddings({
  baseUrl: 'http://localhost:11434', // Default value
  model: 'llama2', // Default value
  requestOptions: {
    useMMap: true, // use_mmap 1
    numThread: 6, // num_thread 6
    numGpu: 1, // num_gpu 1
  },
});

const main = async () => {
  //load youtube video
  // const videoUrls = [
  //   'https://www.youtube.com/watch?v=7Mj3DkHaMUI',
  //   'https://www.youtube.com/watch?v=XB22IOgB3CY',
  //   'https://www.youtube.com/watch?v=TrukLbhLgiw',
  //   'https://www.youtube.com/watch?v=bD6a8_FNumI',
  // ];
  // const loaders = videoUrls.map((video) =>
  //   YoutubeLoader.createFromUrl(video, {
  //     language: 'pt-BR',
  //     addVideoInfo: true,
  //   }).loadAndSplit()
  // );

  // const docs = (await Promise.all(loaders)).flat();
  // console.log('youtube video loaded', docs);
  const docs = [];

  // Create vector store and index the docs
  const vectorStore = await Chroma.fromDocuments(docs, embeddings, {
    collectionName: 'poc-ollama-embeddings',
    url: 'http://localhost:8000', // Optional, will default to this value
    collectionMetadata: {
      'hnsw:space': 'cosine',
    }, // Optional, can be used to specify the distance method of the embedding space https://docs.trychroma.com/usage-guide#changing-the-distance-function
  });

  const retriever = vectorStore.asRetriever();

  // Search for the most similar document
  console.log('docs stored on chroma');

  // text_splitter

  //   const documentEmbeddings = await embeddings.embedDocuments(docs);

  // const prompt = ChatPromptTemplate.fromMessages([
  //   ['system', 'Você é um assistente de busca de conhecumento.'],
  //   [
  //     'human',
  //     'Você vai encontrar respostas para a pergunta dentro da base de conhecimento de vídeos do youtube. Para cada pergunta você deve encontrar as respostas e reponder objetivamente. Pergunta: {question}',
  //   ],
  // ]);

  const prompt = await pull('rlm/rag-prompt', 'pt-BR');

  const outputParser = new StringOutputParser();

  const answerTemplate = 'Aqui está o que eu encontrei. \nResposta: ';
  const answerPrompt = ChatPromptTemplate.fromTemplate(answerTemplate);

  // const chain = prompt
  //   .pipe(model)
  //   .pipe(retriever);

  // const question = 'O que o Bruno Germano acha sobre a Starlink?';
  const question = 'Which camera Bruno is using to make videos?';

  // const response = await chain.invoke({
  //   question,
  // });

  // console.log('response', response, res);

  const template = `Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:`;

  const customRagPrompt = PromptTemplate.fromTemplate(template);

  const ragChain = await createStuffDocumentsChain({
    llm: model,
    prompt: customRagPrompt,
    outputParser: new StringOutputParser(),
  });

  const context = await retriever.getRelevantDocuments(question);

  const response = await ragChain.invoke({
    question,
    context,
  });

  console.log('response', response);
};

main();
