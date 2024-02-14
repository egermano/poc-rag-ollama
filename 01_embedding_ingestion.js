import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { Ollama } from '@langchain/community/llms/ollama';
import { Chroma } from '@langchain/community/vectorstores/chroma';
import { getChannelFeed } from '@obg-lab/youtube-channel-feed';
import dotenv from 'dotenv';
import { YoutubeLoader } from 'langchain/document_loaders/web/youtube';

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
    numThread: 4, // num_thread 6
    numGpu: 1, // num_gpu 1
    mainGpu: 0, // main_gpu 0
  },
});

const getVideos = async () => {
  const channelFeed = await getChannelFeed(process.env.YOUTUBE_CHANNEL_ID);

  const videos = channelFeed.feed.entry.map((entry) => {
    return entry.link[0].$.href;
  });

  return videos;
};

const main = async () => {
  //load youtube video
  const videoUrls = await getVideos();

  const loaders = videoUrls.map((video) =>
    YoutubeLoader.createFromUrl(video, {
      language: 'pt-BR',
      addVideoInfo: true,
    }).loadAndSplit()
  );

  const docs = (await Promise.all(loaders)).flat();
  console.log('youtube video loaded', docs);

  // Create vector store and index the docs
  const vectorStore = await Chroma.fromDocuments(docs, embeddings, {
    collectionName: process.env.CHROMA_COLLECTION,
    url: 'http://localhost:8000', // Optional, will default to this value
    collectionMetadata: {
      'hnsw:space': 'cosine',
    }, // Optional, can be used to specify the distance method of the embedding space https://docs.trychroma.com/usage-guide#changing-the-distance-function
  });

  const retriever = vectorStore.asRetriever();

  // Search for the most similar document
  console.log('docs stored on chroma');

  const question = 'O que o Bruno Germano acha sobre a Starlink?';

  const response = await retriever.invoke(question);

  console.log('response', response);
};

main();
