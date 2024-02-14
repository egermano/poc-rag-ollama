import { Chroma } from '@langchain/community/vectorstores/chroma';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { PromptTemplate } from '@langchain/core/prompts';
import {
  RunnablePassthrough,
  RunnableSequence,
} from '@langchain/core/runnables';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import dotenv from 'dotenv';
import { formatDocumentsAsString } from 'langchain/util/document';

dotenv.config();

const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo',
  temperature: 0,
});

const vectorStore = new Chroma(new OpenAIEmbeddings(), {
  collectionName: process.CHROMA_COLLECTION,
});

const main = async () => {
  const retriever = vectorStore.asRetriever();

  const question = 'Como começar a programar?';

  const template = `Você é um assistente para tarefas de resposta a perguntas. 
    Use as seguintes partes do contexto para responder à pergunta no final.
    Se você não sabe a resposta, apenas diga que não sabe, não tente inventar uma resposta.
    Use no máximo três frases e mantenha a resposta o mais concisa possível.

    Contexto: {context}

    Pergunta: {question}

    Resposta útil baseada no contexto:`;

  const prompt = PromptTemplate.fromTemplate(template);

  const ragChain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocumentsAsString),
      question: new RunnablePassthrough(),
    },
    prompt,
    model,
    new StringOutputParser(),
  ]);

  const response = await ragChain.invoke(question);

  console.log('response', response);
};

main();
