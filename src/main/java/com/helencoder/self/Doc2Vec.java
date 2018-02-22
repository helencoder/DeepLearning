package com.helencoder.self;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;

/**
 * Doc2Vec模型构建
 *
 * Created by helencoder on 2018/2/22.
 */
public class Doc2Vec {

    public static void main(String[] args) {

    }


    /**
     * doc2vec模型训练
     *
     * @param filePath 待训练文本路径
     * @param modelPath  训练模型存储路径
     */
    public void train(String filePath, String modelPath) throws Exception {
        SentenceIterator iter = new BasicLineIterator(new File(filePath));

        AbstractCache<VocabWord> cache = new AbstractCache<>();
        WeightLookupTable<VocabWord> table = new InMemoryLookupTable.Builder<VocabWord>()
                .vectorLength(300)
                .useAdaGrad(false)
                .cache(cache).build();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        LabelsSource source = new LabelsSource("DOC_");
        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(5)
                .epochs(1)
                .layerSize(100)
                .learningRate(0.025)
                .labelsSource(source)
                .windowSize(5)
                .iterate(iter)
                .trainWordVectors(false)
                .vocabCache(cache)
                .lookupTable(table)
                .tokenizerFactory(t)
                .sampling(0)
                .build();

        vec.fit();

        WordVectorSerializer.writeParagraphVectors(vec, modelPath);
    }

    /**
     * doc2vec模型加载
     *
     * @param modelPath 训练模型存储路径
     */
    public ParagraphVectors load(String modelPath) throws Exception {

        ParagraphVectors paragraphVectors = WordVectorSerializer.readParagraphVectors(modelPath);
        return paragraphVectors;
    }

    /**
     * doc2vec模型更新
     *
     * @param filePath 待训练文本路径
     * @param modelPath 训练模型存储路径
     */
    public void update(String filePath, String modelPath) throws Exception {

        ParagraphVectors doc2vec = WordVectorSerializer.readParagraphVectors(modelPath);

        SentenceIterator iterator = new BasicLineIterator(filePath);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        doc2vec.setTokenizerFactory(tokenizerFactory);
        doc2vec.setSentenceIterator(iterator);

        doc2vec.fit();

        WordVectorSerializer.writeParagraphVectors(doc2vec, modelPath);
    }

    /**
     * doc2vec模型应用
     *
     * @param modelPath 训练模型存储路径
     */
    public void apply(String modelPath) throws Exception {
        ParagraphVectors doc2vec = WordVectorSerializer.readParagraphVectors(modelPath);

        // get the boolean of include the doc
        doc2vec.hasWord("doc");

        // get the vector of the doc
        doc2vec.getWordVector("doc");

        // get the similarity between doc1 and doc2
        doc2vec.similarity("doc1", "doc2");

        // get the top 10 closest docs of the doc
        doc2vec.wordsNearest("doc", 10);

    }
}
