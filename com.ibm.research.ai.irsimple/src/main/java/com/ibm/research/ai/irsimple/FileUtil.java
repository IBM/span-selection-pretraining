package com.ibm.research.ai.irsimple;

import java.io.*;
import java.util.*;
import java.util.zip.*;

public class FileUtil {

    private static String GZIP_EXTN = ".gz";

    /**
     * creates any directories that need to exist to create and write the file should not create the
     * file
     */
    public static void ensureWriteable(File f) {
      File parent = f.getParentFile();
      if (parent == null) //has no parents, fine
          return;
      if (!parent.exists() && !parent.mkdirs()) {
        throw new IOError(new IllegalStateException("Couldn't create parent dir: " + parent));
      }
    }
    
    public static Properties loadProperties(String resource) {
        Properties props = new Properties();
        InputStream stream = FileUtil.class.getClassLoader().getResourceAsStream(resource);
        try {
            props.load(stream);
        } catch (IOException ioe) {
            throw new IllegalArgumentException("Unable to load resource" + resource);
        }
        return props;
    }
    
    private static final int BUFFER_SIZE = 2 << 16;

    private static InputStream getInputStream(File file) throws FileNotFoundException, IOException {
        InputStream is = new FileInputStream(file);
        if (file.getAbsolutePath().endsWith(GZIP_EXTN)) {
            return new GZIPInputStream(is, BUFFER_SIZE);
        }
        return new BufferedInputStream(is, BUFFER_SIZE);
    }

    public static class FileIterable implements Iterable<File> {
        File root = null;

        public FileIterable(File root) {
            this.root = root;
        }
        
        @Override
        public Iterator<File> iterator() {
            return new FileIterator(root);
        }
    }

    public static class FileIterator implements Iterator<File> {
        private Queue<File> queue = new LinkedList<File>();
        private File ptr = null;

        public FileIterator(File root) {
            queue.add(root);
        }

        public File peek() {
            if (ptr == null) {
                ptr = next();
            }
            return ptr;
        }

        @Override
        public boolean hasNext() {
            if (ptr != null) {
                return true;
            }
            ptr = next();
            return ptr != null ? true : false;
        }

        @Override
        public File next() {
            if (ptr != null) {
                File file = ptr;
                ptr = null; // reset queue pointer
                return file;
            }
            if (queue.isEmpty()) {
                return null;
            } else {
                File f = queue.remove();
                if (f.isFile()) {
                    return f;
                } else {
                    File[] filesInDir = f.listFiles();
                    if (filesInDir == null) {
                        throw new RuntimeException("Null files when listing: " + f.getAbsolutePath());
                    }
                    for (File file : filesInDir) {
                        queue.add(file);
                    }
                    return next();
                }
            }
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Not Implemented");
        }
    }

    public static class FileLineIterable implements Iterable<String> {
        final File file;
        final boolean skipBlank;

        FileLineIterable(File file, boolean skipBlank) {
            this.file = file;
            this.skipBlank = skipBlank;
        }

        public Iterator<String> iterator() {
            return new FileLineIterator(file, skipBlank);
        }
    }

    public static class FileLineIterator implements Iterator<String> {
        BufferedReader reader;
        String line = null;
        final boolean skipBlank;

        FileLineIterator(File file, boolean skipBlank) {
            this.skipBlank = skipBlank;
            try {
                InputStream is = getInputStream(file);
                reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
            } catch (Exception e) {
                throw new Error(e);
            }
            nextLine();
        }

        private void nextLine() {
            try {
                if (reader == null)
                    return;
                line = reader.readLine();
                if (line == null) {
                    reader.close();
                    reader = null;
                }
            } catch (Exception e) {
                throw new Error(e);
            }
            if (skipBlank && line != null && line.length() == 0) {
                nextLine();
            }
        }

        public boolean hasNext() {
            return line != null;
        }

        public String next() {
            String curLine = line;
            nextLine();
            return curLine;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

    }
}
