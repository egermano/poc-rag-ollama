import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { Ollama } from '@langchain/community/llms/ollama';
import { Chroma } from '@langchain/community/vectorstores/chroma';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { PromptTemplate } from '@langchain/core/prompts';
import {
  RunnableMap,
  RunnablePassthrough,
  RunnableSequence,
} from '@langchain/core/runnables';
import dotenv from 'dotenv';

dotenv.config();

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

const vectorStore = new Chroma(embeddings, {
  collectionName: process.env.CHROMA_COLLECTION,
  url: 'http://localhost:8000', // Optional, will default to this value
  collectionMetadata: {
    'hnsw:space': 'cosine',
  }, // Optional, can be used to specify the distance method of the embedding space https://docs.trychroma.com/usage-guide#changing-the-distance-function
});

const main = async () => {
  const retriever = vectorStore.asRetriever({
    k: 5,
    verbose: true,
    searchType: 'similarity',
    metadata: true,
  });

  const template = `Você é um assistente para tarefas de resposta a perguntas. 
    Use as seguintes partes do contexto para responder à pergunta no final.
    Se você não sabe a resposta, apenas diga que não sabe, não tente inventar uma resposta.
    Use no máximo três frases e mantenha a resposta o mais concisa possível.

    Contexto: {context}

    Pergunta: {question}

    Resposta útil baseada no contexto:`;

  const prompt = PromptTemplate.fromTemplate(template);

  const ragChainFromDocs = RunnableSequence.from(
    [
      RunnablePassthrough.assign({
        context: (input) => {
          return input.context.map(
            (doc) =>
              `content: ${doc.pageContent} \n metadata: ${JSON.stringify(
                doc.metadata
              )}`
          );
        },
      }),
      prompt,
      model,
      new StringOutputParser(),
    ],
    'ragChainFromDocs'
  );

  let ragChainWithSource = new RunnableMap({
    steps: { context: retriever, question: new RunnablePassthrough() },
  });

  ragChainWithSource = ragChainWithSource.assign({ answer: ragChainFromDocs });

  const question = 'Quais equipamentos o Bruno Germano usa para viajar e trabalhar?';

  const response = await ragChainWithSource.invoke(question);

  console.log('response', response);
};

main();
