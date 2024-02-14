import { Ollama } from '@langchain/community/llms/ollama';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';

const model = new Ollama({
  baseUrl: 'http://localhost:11434', // Default value
  model: 'llama2', // Default value
});

const prompt = ChatPromptTemplate.fromMessages([
  ['human', 'Answer the questions in the same language. Question: {question}'],
]);
const outputParser = new StringOutputParser();

const chain = prompt.pipe(model).pipe(outputParser);

chain.invoke({ question: 'O que é a Starlink?' }).then((res) => {
  console.log(res);
});
