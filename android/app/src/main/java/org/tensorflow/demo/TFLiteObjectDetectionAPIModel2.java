
package org.tensorflow.demo;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.StringTokenizer;
import java.util.Vector;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.lite.Interpreter;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteObjectDetectionAPIModel2 implements Classifier {
  private static final Logger LOGGER = new Logger();

  // Only return this many results.
  private static final int NUM_RESULTS = 1917;
  private static final int NUM_CLASSES = 91;

  private static final float Y_SCALE = 10.0f;
  private static final float X_SCALE = 10.0f;
  private static final float H_SCALE = 5.0f;
  private static final float W_SCALE = 5.0f;

  // Config values.
  private int inputSize;

  private final float[][] boxPriors = new float[4][NUM_RESULTS];

  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  private float[][][][] outputLocations;
  private float[][][] outputClasses;

  private byte[][][][] outputLocations_;
  private byte[][][] outputClasses_;

  private boolean isModelQuantized;

  float[][][][] img;
  protected ByteBuffer imgData = null;


  private Interpreter tfLite;

  private float expit(final float x) {
    return (float) (1. / (1. + Math.exp(-x)));
  }

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
          throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  private void loadCoderOptions(
          final AssetManager assetManager, final String locationFilename, final float[][] boxPriors)
          throws IOException {
    // Try to be intelligent about opening from assets or sdcard depending on prefix.
    final String assetPrefix = "file:///android_asset/";
    InputStream is;
    if (locationFilename.startsWith(assetPrefix)) {
      is = assetManager.open(locationFilename.split(assetPrefix, -1)[1]);
    } else {
      is = new FileInputStream(locationFilename);
    }

    final BufferedReader reader = new BufferedReader(new InputStreamReader(is));

    for (int lineNum = 0; lineNum < 4; ++lineNum) {
      String line = reader.readLine();
      final StringTokenizer st = new StringTokenizer(line, ", ");
      int priorIndex = 0;
      while (st.hasMoreTokens()) {
        final String token = st.nextToken();
        try {
          final float number = Float.parseFloat(token);
          boxPriors[lineNum][priorIndex++] = number;
        } catch (final NumberFormatException e) {
          // Silently ignore.
        }
      }
      if (priorIndex != NUM_RESULTS) {
        throw new RuntimeException(
                "BoxPrior length mismatch: " + priorIndex + " vs " + NUM_RESULTS);
      }
    }

    LOGGER.i("Loaded box priors!");
  }

  void decodeCenterSizeBoxes(float[][][][] predictions) {
    for (int i = 0; i < NUM_RESULTS; ++i) {
      float ycenter = predictions[0][i][0][0] / Y_SCALE * boxPriors[2][i] + boxPriors[0][i];
      float xcenter = predictions[0][i][0][1] / X_SCALE * boxPriors[3][i] + boxPriors[1][i];
      float h = (float) Math.exp(predictions[0][i][0][2] / H_SCALE) * boxPriors[2][i];
      float w = (float) Math.exp(predictions[0][i][0][3] / W_SCALE) * boxPriors[3][i];

      float ymin = ycenter - h / 2.f;
      float xmin = xcenter - w / 2.f;
      float ymax = ycenter + h / 2.f;
      float xmax = xcenter + w / 2.f;

      predictions[0][i][0][0] = ymin;
      predictions[0][i][0][1] = xmin;
      predictions[0][i][0][2] = ymax;
      predictions[0][i][0][3] = xmax;
    }
  }

  void decodeCenterSizeBoxes2(byte[][][][] predictions) {
    for (int i = 0; i < NUM_RESULTS; ++i) {
      float ycenter = ((predictions[0][i][0][0] & 0xff) / 255.0f) / Y_SCALE * boxPriors[2][i] + boxPriors[0][i];
      float xcenter = ((predictions[0][i][0][1] & 0xff) / 255.0f) / X_SCALE * boxPriors[3][i] + boxPriors[1][i];
      float h = (float) Math.exp(((predictions[0][i][0][2] & 0xff) / 255.0f) / H_SCALE) * boxPriors[2][i];
      float w = (float) Math.exp(((predictions[0][i][0][3] & 0xff) / 255.0f) / W_SCALE) * boxPriors[3][i];

      float ymin = ycenter - h / 2.f;
      float xmin = xcenter - w / 2.f;
      float ymax = ycenter + h / 2.f;
      float xmax = xcenter + w / 2.f;

      predictions[0][i][0][0] = (byte) ((int)ymin & 0xFF);
      predictions[0][i][0][1] = (byte) ((int)xmin & 0xFF);
      predictions[0][i][0][2] = (byte) ((int)ymax & 0xFF);
      predictions[0][i][0][3] = (byte) ((int)xmax & 0xFF);
    }
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   */
  public static Classifier create(
          final AssetManager assetManager,
          final String modelFilename,
          final String labelFilename,
          final int inputSize,
          final boolean isQuantized) throws IOException {
    final TFLiteObjectDetectionAPIModel2 d = new TFLiteObjectDetectionAPIModel2();

    d.loadCoderOptions(assetManager, "file:///android_asset/box_priors.txt", d.boxPriors);

    InputStream labelsInput = null;
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    labelsInput = assetManager.open(actualFilename);
    BufferedReader br = null;
    br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();

    d.inputSize = inputSize;

    try {
      d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    // Pre-allocate buffers.
    d.img = new float[1][inputSize][inputSize][3];

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());

    d.intValues = new int[d.inputSize * d.inputSize];
    d.outputLocations = new float[1][NUM_RESULTS][1][4];
    d.outputClasses = new float[1][NUM_RESULTS][NUM_CLASSES];
    return d;
  }

  private TFLiteObjectDetectionAPIModel2() {}

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixel = intValues[j * inputSize + i];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixel >> 16) & 0xFF));
          imgData.put((byte) ((pixel >> 8) & 0xFF));
          imgData.put((byte) (pixel & 0xFF));
        } else {
          img[0][j][i][2] = (float) (pixel & 0xFF) / 128.0f - 1.0f;
          img[0][j][i][1] = (float) ((pixel >> 8) & 0xFF) / 128.0f - 1.0f;
          img[0][j][i][0] = (float) ((pixel >> 16) & 0xFF) / 128.0f - 1.0f;
        }
      }
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
    if(isModelQuantized)
    {
      outputLocations_ = new byte[1][NUM_RESULTS][1][4];
      outputClasses_ = new byte[1][NUM_RESULTS][NUM_CLASSES];

      Object[] inputArray = {imgData};
      Map<Integer, Object> outputMap = new HashMap<>();
      outputMap.put(0, outputLocations_);
      outputMap.put(1, outputClasses_);
      Trace.endSection();

      // Run the inference call.
      Trace.beginSection("run");
      tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
      Trace.endSection();

      decodeCenterSizeBoxes2(outputLocations_);

      // Find the best detections.
      final PriorityQueue<Recognition> pq =
              new PriorityQueue<Recognition>(
                      1,
                      new Comparator<Recognition>() {
                        @Override
                        public int compare(final Recognition lhs, final Recognition rhs) {
                          // Intentionally reversed to put high confidence at the head of the queue.
                          return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                        }
                      });

      // Scale them back to the input size.
      for (int i = 0; i < NUM_RESULTS; ++i) {
        float topClassScore = -1000f;
        int topClassScoreIndex = -1;

        // Skip the first catch-all class.
        for (int j = 1; j < NUM_CLASSES; ++j) {
          float score = expit(((outputClasses_[0][i][j] & 0xff) / 255.0f));
          LOGGER.d("Byte score: %X, Float score: %f", outputClasses_[0][i][j], score);

          if (score > topClassScore) {
            topClassScoreIndex = j;
            topClassScore = score;
          }
        }

        if (topClassScore > 0.001f) {
          final RectF detection =
                  new RectF(
                          ((outputLocations_[0][i][0][1] & 0xff) / 255.0f) * inputSize,
                          ((outputLocations_[0][i][0][0] & 0xff) / 255.0f) * inputSize,
                          ((outputLocations_[0][i][0][3] & 0xff) / 255.0f) * inputSize,
                          ((outputLocations_[0][i][0][2] & 0xff) / 255.0f) * inputSize);

          pq.add(
                  new Recognition(
                          "" + i,
                          labels.get(topClassScoreIndex),
                          ((outputClasses_[0][i][topClassScoreIndex] & 0xff) / 255.0f),
                          detection));
        }
      }
      for (int i = 0; i < Math.min(pq.size(), 10); ++i) {
        Recognition recog = pq.poll();
        recognitions.add(recog);
      }
      Trace.endSection(); // "recognizeImage"
    }
    else{
      outputLocations = new float[1][NUM_RESULTS][1][4];
      outputClasses = new float[1][NUM_RESULTS][NUM_CLASSES];

      Object[] inputArray = {img};
      Map<Integer, Object> outputMap = new HashMap<>();
      outputMap.put(0, outputLocations);
      outputMap.put(1, outputClasses);
      Trace.endSection();

      // Run the inference call.
      Trace.beginSection("run");
      tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
      Trace.endSection();

      decodeCenterSizeBoxes(outputLocations);

      // Find the best detections.
      final PriorityQueue<Recognition> pq =
              new PriorityQueue<Recognition>(
                      1,
                      new Comparator<Recognition>() {
                        @Override
                        public int compare(final Recognition lhs, final Recognition rhs) {
                          // Intentionally reversed to put high confidence at the head of the queue.
                          return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                        }
                      });

      // Scale them back to the input size.
      for (int i = 0; i < NUM_RESULTS; ++i) {
        float topClassScore = -1000f;
        int topClassScoreIndex = -1;

        // Skip the first catch-all class.
        for (int j = 1; j < NUM_CLASSES; ++j) {
          float score = expit(outputClasses[0][i][j]);

          if (score > topClassScore) {
            topClassScoreIndex = j;
            topClassScore = score;
          }
        }

        if (topClassScore > 0.001f) {
          final RectF detection =
                  new RectF(
                          outputLocations[0][i][0][1] * inputSize,
                          outputLocations[0][i][0][0] * inputSize,
                          outputLocations[0][i][0][3] * inputSize,
                          outputLocations[0][i][0][2] * inputSize);

          pq.add(
                  new Recognition(
                          "" + i,
                          labels.get(topClassScoreIndex),
                          outputClasses[0][i][topClassScoreIndex],
                          detection));
        }
      }
      for (int i = 0; i < Math.min(pq.size(), 10); ++i) {
        Recognition recog = pq.poll();
        recognitions.add(recog);
      }
      Trace.endSection(); // "recognizeImage"
    }


    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {
  }

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {
  }
}