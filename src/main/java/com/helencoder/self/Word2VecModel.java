package com.helencoder.self;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;


/**
 * Word2Vec模型构建
 *
 * dependencies: deeplearning4j-core、deeplearning4j-nlp
 *
 * Created by helencoder on 2018/2/22.
 */
public class Word2VecModel {

    /**
     * word2vec模型训练
     *
     * @param filePath 待训练文本路径
     * @param modelPath  训练模型存储路径
     */
    public void train(String filePath, String modelPath) throws Exception {

        SentenceIterator iter = new BasicLineIterator(filePath);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        VocabCache<VocabWord> cache = new AbstractCache<>();
        WeightLookupTable<VocabWord> table = new InMemoryLookupTable.Builder<VocabWord>()
                .vectorLength(200)
                .useAdaGrad(false)
                .cache(cache).build();

        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .epochs(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .lookupTable(table)
                .vocabCache(cache)
                .build();

        vec.fit();

        WordVectorSerializer.writeWord2VecModel(vec, modelPath);
    }

    /**
     * word2vec模型加载
     *
     * @param modelPath 训练模型存储路径
     */
    public Word2Vec load(String modelPath) {

        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(modelPath);
        return word2Vec;
    }

    /**
     * word2vec模型更新
     *
     * @param filePath 待训练文本路径
     * @param modelPath 训练模型存储路径
     */
    public void update(String filePath, String modelPath) throws Exception {

        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(modelPath);

        SentenceIterator iterator = new BasicLineIterator(filePath);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        word2Vec.setTokenizerFactory(tokenizerFactory);
        word2Vec.setSentenceIterator(iterator);

        word2Vec.fit();

        WordVectorSerializer.writeWord2VecModel(word2Vec, modelPath);
    }

    /**
     * word2vec模型应用
     *
     * @param modelPath 训练模型存储路径
     */
    public void apply(String modelPath) {
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(modelPath);

        // get the boolean of include the word
        word2Vec.hasWord("word");

        // get the vector of the word
        word2Vec.getWordVector("word");

        // get the similarity between word1 and word2
        word2Vec.similarity("word1", "word2");

        // get the top 10 closest words of the word
        word2Vec.wordsNearest("word", 10);
    }

}
