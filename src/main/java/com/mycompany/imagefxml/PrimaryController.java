package com.mycompany.imagefxml;

import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.ListView;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.TextAlignment;
import javafx.scene.transform.Rotate;
import javafx.stage.FileChooser;
import javax.imageio.ImageIO;

import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

public class PrimaryController {

    @FXML
    private ImageView imageView;
    private Image image;
    private double ratio = 1;
    private Rotate rotate = new Rotate();

    @FXML
    private Rectangle rectangle;
    private double startX, startY;

    @FXML
    private ListView<String> imageListView;

    private List<String> imagePaths;

    private int IMG_W = 700;
    private int IMG_H = 700;
    private int gridWidth = 15;
    private int gridHeight = 15;
    private float threshold = 0.26f;

    private String imageURL;
    private File imageFile;

    private String[] CLASSES = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"};

    private String[] COLORS = {"#6793be", "#00ffaa", "#fececf", "#ffbcc9", "#ffb9c7", "#fdc6d1",
        "#fdc9d3", "#6793be", "#73a4d4", "#9abde0", "#9abde0", "#8fff8f", "#ffcfd8", "#808080", "#808080",
        "#ffba00", "#6699ff", "#009933", "#1c1c1c", "#08375f", "#116ebf", "#e61d35", "#106bff", "#8f8fff",
        "#8fff8f", "#dbdbff", "#dbffdb", "#dbffff", "#ffdbdb", "#ffc2c2", "#ffa8a8", "#ff8f8f", "#e85e68",
        "#123456", "#5cd38c", "#1d1f5f", "#4e4b04", "#495a5b", "#489d73", "#9d4872", "#d49ea6", "#ff0080"};

    private Canvas canvas = new Canvas(IMG_W, IMG_H);

    @FXML
    private void onLoad(ActionEvent event) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("JPG Files", "*.jpg", "*.jpeg"),
                new FileChooser.ExtensionFilter("GIF Files", "*.gif"),
                new FileChooser.ExtensionFilter("PNG Files", "*.png"),
                new FileChooser.ExtensionFilter("BMP Files", "*.bmp")
        );
        File file = fileChooser.showOpenDialog(null);
        if (file != null) {
            String imagePath = file.getAbsolutePath();
            imagePaths.add(imagePath);
            imageListView.getItems().add(imagePath);
        }
    }

    @FXML
    public void showImage() {
        String selectedImagePath = imageListView.getSelectionModel().getSelectedItem();
        if (selectedImagePath != null) {
            image = new Image(selectedImagePath);
            imageView.setImage(image);
            imageURL = image.getUrl();
            imageFile = new File(imageURL);
            resetRatio();
        }
    }

    public void initialize() {
        imagePaths = new ArrayList<>();
        imagePaths.add("https://static.wikia.nocookie.net/1bde55f8-3077-41bc-afd0-dca65c09853d");
        imagePaths.add("https://www.vmcdn.ca/f/files/victoriatimescolonist/json/2022/03/web1_vka-viewstreet-13264.jpg;w=960");
        imagePaths.add("https://tamashiiweb.com/images/item/item_0000013411_RoemZgLS_04.jpg");
        imageListView.getItems().addAll(imagePaths);
        imageView.setOnMousePressed(this::handleMousePressed);
        imageView.setOnMouseDragged(this::handleMouseDragged);
        imageView.setOnMouseReleased(this::handleMouseReleased);
        rectangle.setVisible(false);
    }

    @FXML
    void onSave(ActionEvent event) throws IOException {
        String selectedImagePath = imageListView.getSelectionModel().getSelectedItem();
        if (selectedImagePath != null) {
            Image image = imageView.getImage();
            FileChooser fileChooser = new FileChooser();

            FileChooser.ExtensionFilter imageFilter = new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp");
            fileChooser.getExtensionFilters().add(imageFilter);

            File outputFile = fileChooser.showSaveDialog(null);
            if (outputFile != null) {
                try {
                    String fileExtension = getFileExtension(outputFile.getName());
                    WritableImage wImage = imageView.snapshot(null, null);
                    BufferedImage bufferedImage = SwingFXUtils.fromFXImage(wImage, null);
                    ImageIO.write(bufferedImage, fileExtension, outputFile);
                    System.out.println("Image saved successfully.");
                } catch (IOException e) {
                    System.out.println("Error saving image: " + e.getMessage());
                }
            }
        }
    }

    private String getFileExtension(String fileName) {
        int dotIndex = fileName.lastIndexOf('.');
        if (dotIndex > 0 && dotIndex < fileName.length() - 1) {
            return fileName.substring(dotIndex + 1).toLowerCase();
        }
        return "";
    }

    @FXML
    void onRotate90(ActionEvent event) {
        rotate.setAngle(90);
        rotate.setPivotX(imageView.getFitWidth() / 2);
        rotate.setPivotY(imageView.getFitHeight() / 2);
        imageView.getTransforms().add(rotate);
    }

    @FXML
    void onRotateNeg90(ActionEvent event) {
        rotate.setAngle(-90);
        rotate.setPivotX(imageView.getFitWidth() / 2);
        rotate.setPivotY(imageView.getFitHeight() / 2);
        imageView.getTransforms().add(rotate);
    }

    @FXML
    public void zoomIn() {
        ratio *= 1.1;
        resizeImage();
    }

    @FXML
    public void zoomOut() {
        ratio /= 1.1;
        resizeImage();
    }

    private void resizeImage() {
        imageView.setFitWidth(image.getWidth() * ratio);
        imageView.setFitHeight(image.getHeight() * ratio);
    }

    private void resetRatio() {
        ratio = 1.0;
        resizeImage();
    }

    @FXML
    public void onFlipVertical() {
        if (image != null) {
            WritableImage wImage = imageView.snapshot(null, null);
            BufferedImage bufferedImage = SwingFXUtils.fromFXImage(wImage, null);
            BufferedImage flippedImage = flipImage(bufferedImage, true);
            Image flippedFXImage = SwingFXUtils.toFXImage(flippedImage, null);
            imageView.setImage(flippedFXImage);
        }
    }

    @FXML
    public void onFlipHorizontal() {
        if (image != null) {
            WritableImage wImage = imageView.snapshot(null, null);
            BufferedImage bufferedImage = SwingFXUtils.fromFXImage(wImage, null);
            BufferedImage flippedImage = flipImage(bufferedImage, false);
            Image flippedFXImage = SwingFXUtils.toFXImage(flippedImage, null);
            imageView.setImage(flippedFXImage);
        }
    }

    private BufferedImage flipImage(BufferedImage image, boolean horizontal) {
        int width = image.getWidth();
        int height = image.getHeight();

        BufferedImage flippedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int srcX = horizontal ? width - x - 1 : x;
                int srcY = horizontal ? y : height - y - 1;
                flippedImage.setRGB(x, y, image.getRGB(srcX, srcY));
            }
        }

        return flippedImage;
    }

    private void handleMousePressed(MouseEvent event) {
        System.out.println("Pressed");
        startX = event.getX();
        startY = event.getY();

        // Set the initial position of the rectangle
        rectangle.setX(startX);
        rectangle.setY(startY);
        rectangle.setVisible(true);

    }

    private void handleMouseDragged(MouseEvent event) {
        double currentX = event.getX();
        double currentY = event.getY();

        // Calculate the width and height of the rectangle
        double width = currentX - startX;
        double height = currentY - startY;

        // Update the position and size of the rectangle
        rectangle.setWidth(Math.abs(width));
        rectangle.setHeight(Math.abs(height));

        rectangle.setWidth(Math.abs(width));
        rectangle.setHeight(Math.abs(height));
        rectangle.setX(width < 0 ? currentX : startX);
        rectangle.setY(height < 0 ? currentY : startY);
    }

    private void handleMouseReleased(MouseEvent event) {
        double scaleFactor = imageView.getImage().getWidth() / imageView.getFitWidth();
        double cropX = rectangle.getX() * scaleFactor;
        double cropY = rectangle.getY() * scaleFactor;
        double cropWidth = rectangle.getWidth() * scaleFactor;
        double cropHeight = rectangle.getHeight() * scaleFactor;

        WritableImage writableImage = new WritableImage(
                imageView.getImage().getPixelReader(),
                (int) cropX,
                (int) cropY,
                (int) cropWidth,
                (int) cropHeight
        );
        RenderedImage renderedImage = SwingFXUtils.fromFXImage(writableImage, null);
        FileChooser fileChooser = new FileChooser();
        FileChooser.ExtensionFilter imageFilter = new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp");
        fileChooser.getExtensionFilters().add(imageFilter);
        File outputFile = fileChooser.showSaveDialog(null);

        try {
            String fileExtension = getFileExtension(outputFile.getName());
            ImageIO.write(renderedImage, fileExtension, outputFile);

            System.out.println("Cropped image saved to: " + outputFile.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @FXML
    public void onDetect() throws Exception {
        GraphicsContext ctx = canvas.getGraphicsContext2D();
        ctx.drawImage(new Image(new FileInputStream(imageFile), IMG_W, IMG_H, false, false), 0, 0);
        runYOLO(ctx);

    }

    private INDArray loadImage() throws IOException {
        NativeImageLoader imageLoader = new NativeImageLoader(IMG_W, IMG_H, 3,
                new ColorConversionTransform(org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2RGB));
        INDArray image = imageLoader.asMatrix(imageFile);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        return image;
    }

    public void runYOLO(GraphicsContext ctx) throws IOException {
        ComputationGraph yoloModel = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
        System.out.println("Model: " + yoloModel.summary());

        INDArray loadedImage = loadImage();
        INDArray output = yoloModel.outputSingle(loadedImage);
        Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) yoloModel.getOutputLayer(0);
        List<DetectedObject> predictedObjects = outputLayer.getPredictedObjects(output, threshold);
        System.out.println("Number of predicted objects: " + predictedObjects.size());

        int w = IMG_W;
        int h = IMG_H;

        Map<String, Paint> colors = new HashMap();

        for (int i = 0; i < CLASSES.length; i++) {
            colors.put(CLASSES[i], Color.web(COLORS[i]));
        }

        ctx.setLineWidth(3);
        ctx.setTextAlign(TextAlignment.CENTER);
        for (DetectedObject obj : predictedObjects) {
            if (obj.getPredictedClass() < 20) {
                String cl = CLASSES[obj.getPredictedClass()];
                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();
                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                int y2 = (int) Math.round(h * xy2[1] / gridHeight);
                int rectW = x2 - x1;
                int rectH = y2 - y1;
                System.out.printf("%s - %d, %d, %d, %d \n", cl, x1, x2, y1, y2);
                ctx.setStroke(colors.get(cl));
                ctx.strokeRect(x1, y1, rectW, rectH);
                ctx.strokeText(cl, x1 + (rectW / 2), y1 - 2);
                ctx.setFill(Color.WHITE);
                ctx.fillText(cl, x1 + (rectW / 2), y1 - 2);
            }
        }
        image = canvas.snapshot(null, null);
        imageView.setImage(image);
    }
}

//Resize, crop, flip, rotate img, object detection
