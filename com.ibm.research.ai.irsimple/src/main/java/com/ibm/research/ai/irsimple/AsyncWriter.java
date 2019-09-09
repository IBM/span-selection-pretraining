package com.ibm.research.ai.irsimple;

import java.io.*;
import java.util.*;
import java.util.zip.*;

import com.google.gson.*;

public class AsyncWriter {
    
    public static class Query {
        public String sentence;
        public String docId;
        public String getSearchQuery() {
            return sentence.substring(0, sentence.indexOf('▁')) + sentence.substring(sentence.lastIndexOf('▁')+1);
        }
        public String getAnswer() {
            return sentence.substring(sentence.indexOf('▁')+1, sentence.lastIndexOf('▁'));
        }
        public String getSSPTQuery() {
            return sentence.substring(0, sentence.indexOf('▁')) + " [BLANK] " + sentence.substring(sentence.lastIndexOf('▁')+1);
        }
    }
    
    public static class SSPT {
        public String qid;
        public String question;
        public String passage;
        public String[] answers;
    }
    
    public static File getFile(File workingDir, List<File> current) {
        while (!workingDir.exists()) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {}
        }
        Set<String> names = new HashSet<>();
        for (File f : current)
            names.add(f.getName());
        boolean finished = false;
        while (!finished) {
            for (File f : workingDir.listFiles()) {
                if (f.getName().equals("finished.txt")) {
                    finished = true;
                }
                if (f.getName().startsWith("queries") && f.getName().endsWith(".jsonl.gz") && !names.contains(f.getName())) {
                    return f;
                }
            }
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {}
        }
        return null;
    }
    
    public static class ShuffledQueries {
        File workingDir;
        List<Iterator<String>> lineIterators;
        List<File> files;
        int maxOpenFiles = 20;  //how many query files do we shuffle together
        boolean finished = false;  //no more queries* files will be created
        Random rand = new Random();
        public ShuffledQueries(File workingDir) {
            this.workingDir = workingDir;
            this.lineIterators = new ArrayList<>();
            this.files = new ArrayList<>();
        }
        
        public String getLine() {
            while (!finished && lineIterators.size() < maxOpenFiles) {
                File f = getFile(workingDir, files);
                if (f == null) {
                    finished = true;
                } else {
                    lineIterators.add(new FileUtil.FileLineIterator(f, true));
                    files.add(f);
                }
            }
            if (lineIterators.isEmpty())
                return null;
            int fndx = rand.nextInt(lineIterators.size());
            if (!lineIterators.get(fndx).hasNext()) {
                lineIterators.remove(fndx);
                System.err.println("Finished "+files.get(fndx).getAbsolutePath());
                files.remove(fndx).delete();
                return getLine();
            }
            return lineIterators.get(fndx).next();
        }
    }
    
    public static class Writer {
        String baseDir = null;
        int dirNdx = 0;
        int fileNdx = 0;
        File currentFile = null;
        PrintStream out = null;
        int fileInstCount = 0;
        
        final int fileInstLimit = 100000;
        final int dirFileLimit = 100;
        
        public Writer(File writeDir) {
            this.baseDir = writeDir.getAbsolutePath();
        }
        
        private void setupPrintStream() {
            if (fileInstCount >= fileInstLimit) {
                // close outputstream
                out.close();
                out = null;
                // rename from .partial
                String fname = currentFile.getAbsolutePath();
                fname = fname.substring(0, fname.length()-".partial".length());
                currentFile.renameTo(new File(fname));
                currentFile = null;
                // update file and dir counters
                fileInstCount = 0;
                fileNdx += 1;
                if (fileNdx >= dirFileLimit) {
                    dirNdx += 1;
                    fileNdx = 0;
                }
            }
            if (out == null) {
                try {
                    String fname = baseDir + "/" + String.format("%03d", dirNdx) + 
                                   "/sspt_" + String.format("%03d", fileNdx) + ".jsonl.gz.partial";
                    currentFile = new File(fname);
                    FileUtil.ensureWriteable(currentFile);
                    OutputStream os = new GZIPOutputStream(new FileOutputStream(currentFile));
                    out = new PrintStream(os, false, "UTF-8");
                } catch (Exception e) {
                    throw new Error(e);
                }
            }
        }
        
        public synchronized void write(String json) {
            //write to file, switch to another file if it gets too big
            setupPrintStream();
            out.println(json);
            fileInstCount += 1;
        }
        
        public void close() {
            if (out != null) {
                out.close();
            }
            // delete any .partial, we only want full-sized file
            if (currentFile != null) {
                System.err.println("Removing partially complete file: "+currentFile.getAbsolutePath());
                currentFile.delete();
            }
        }
    }
    
    /**
     * index_dir is created from doc_dir by:
     *  java -cp irsimple.jar com.ibm.research.ai.irsimple.MakeIndex yourDir/wikipassagesdir yourDir/wikipassagesindex
     *
     * Example args:
     *  yourDir/ssptGen yourDir/wikipassagesindex
     * 
     * read query*.jsonl.gz from workingDir (the first arg)
     * write sspt*.jsonl.gz to the same directory (or multiple subdirectories)
     * when we see finished.txt and no query*.jsonl.gz we know the query generation is done
     * @param args
     */
    public static void main(String[] args) {
        File workingDir = new File(args[0]);
        String indexDir = args[1];
        
        int topN = 20;
        float minScore = 25;
        float maxScore = 70;
        float targetNoAnswerFraction = 0.3f;
        
        Search search = new Search(indexDir, topN, minScore, maxScore, targetNoAnswerFraction);
        Writer writer = new Writer(workingDir);
        PollingThreadedExecutor threads = new PollingThreadedExecutor();
        Gson gson = new Gson();
        long startTime = System.currentTimeMillis();
        
        int qCount = 0;
        String linetmp = null;
        ShuffledQueries queryLines = new ShuffledQueries(workingDir);
        while ((linetmp = queryLines.getLine()) != null) {
            String line = linetmp;
            threads.execute(() -> {
                Query q = gson.fromJson(line, Query.class);
                String query = q.getSearchQuery();
                String answer = q.getAnswer();
                String passage = search.query(query, q.docId, answer);
                if (passage != null) {
                    SSPT inst = new SSPT();
                    inst.qid = UUID.randomUUID().toString();
                    inst.question = q.getSSPTQuery();
                    inst.passage = passage;
                    inst.answers = new String[] {answer};
                    String jsonLine = gson.toJson(inst);
                    writer.write(jsonLine);
                }
            });
            qCount += 1;
            if (qCount % 10000 == 0) {
                System.err.println("Recall "+(double)search.hasAnswerCount/qCount+" "+
                                   1000.0*qCount/(System.currentTimeMillis()-startTime)+" queries per second");
            }
        }
        threads.awaitFinishing();
        writer.close();
    }
}
