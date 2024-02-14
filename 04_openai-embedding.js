import { Chroma } from '@langchain/community/vectorstores/chroma';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { getChannelFeed } from '@obg-lab/youtube-channel-feed';
import dotenv from 'dotenv';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { YoutubeLoader } from 'langchain/document_loaders/web/youtube';
import { pull } from 'langchain/hub';

dotenv.config();

const getVideos = async () => {
  const channelFeed = await getChannelFeed(process.env.YOUTUBE_CHANNEL_ID);

  const videos = channelFeed.feed.entry.map((entry) => {
    return entry.link[0].$.href;
  });

  return videos;
};

const main = async () => {
  const videoUrls = await getVideos();

  const loaders = videoUrls.map((video) =>
    YoutubeLoader.createFromUrl(video, {
      language: 'pt-BR',
      addVideoInfo: true,
    }).loadAndSplit()
  );

  const docs = (await Promise.all(loaders)).flat();

  const vectorStore = await Chroma.fromDocuments(docs, new OpenAIEmbeddings(), {
    collectionName: process.env.CHROMA_COLLECTION,
    url: 'http://localhost:8000', // Optional, will default to this value
    collectionMetadata: {
      'hnsw:space': 'cosine',
    }, // Optional, can be used to specify the distance method of the embedding space https://docs.trychroma.com/usage-guide#changing-the-distance-function
  });

  // Retrieve and generate using the relevant snippets of the blog.
  const retriever = vectorStore.asRetriever();
  const prompt = await pull('rlm/rag-prompt');
  const llm = new ChatOpenAI({ modelName: 'gpt-3.5-turbo', temperature: 0 });

  const ragChain = await createStuffDocumentsChain({
    llm,
    prompt,
    outputParser: new StringOutputParser(),
  });

  const retrievedDocs = await retriever.getRelevantDocuments(
    'what is task decomposition'
  );

  const response = await ragChain.invoke({
    question: 'What is task decomposition?',
    context: retrievedDocs,
  });

  console.log('response', response);
};

main();
