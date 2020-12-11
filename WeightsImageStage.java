package org.deeplearning4j.examples.wip.javafxui;

import javafx.geometry.Point3D;
import javafx.scene.*;
import javafx.scene.control.Tooltip;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.scene.transform.Rotate;
import javafx.scene.transform.Translate;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import javafx.util.Duration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Visualization of layer weights (parameters).  Ctrl-click and drag on an image to move it.
 * Navigate with arrow keys.  Click and drag to rotate (ctrl for rotation around x axis).
 * @author Don Smith  (ThinkerFeeler@gmail.com)
 */
public class WeightsImageStage extends Stage {
    private static final int width = 1200;
    private static final int height = 800;
    private static final Point3D YAXIS = new Point3D(0, 1, 0);
    private final MultiLayerNetwork net;
    private final List<ImageView> imageViews = new ArrayList<>();
    private final Group root = new Group();
    private final Scene scene;
    private final XformWorld world = new XformWorld();
    private double mousePosX, mousePosY, mouseOldX, mouseOldY, mouseDeltaX, mouseDeltaY;
    private final PerspectiveCamera camera = new PerspectiveCamera(true);
    private final XformCamera cameraXform = new XformCamera();
    private static final double CAMERA_NEAR_CLIP = 0.1;
    private static final double CAMERA_FAR_CLIP = 20000.0;
    private static double cameraInitialX = 100;
    private static double cameraInitialY = 930;
    private double cameraInitialZ = -4000;
    private static final DateTimeFormatter dateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss");
    private ImageView selectedImageView = null;
    // -------------------------
    private static class XformWorld extends Group {
        final Translate t = new Translate(0.0, 0.0, 0.0);
        final Rotate rx = new Rotate(0,width/2, height/2, 0, Rotate.X_AXIS);
        final Rotate ry = new Rotate(180,width/2, height/2, 0, Rotate.Y_AXIS);
        final Rotate rz = new Rotate(0,width/2, height/2, 0, Rotate.Z_AXIS);

        public XformWorld() {
            super();
            this.getTransforms().addAll(t, rx, ry, rz);
        }
    }
    //------------------------------

    // -------------------------
    private static class XformCamera extends Group {
        final Translate t = new Translate(0.0, 0.0, 0.0);
        final Rotate rx = new Rotate(0, 0, 0, 0, Rotate.X_AXIS);
        final Rotate ry = new Rotate(0, 0, 0, 0, Rotate.Y_AXIS);
        final Rotate rz = new Rotate(0, 0, 0, 0, Rotate.Z_AXIS);

        public XformCamera() {
            super();
            this.getTransforms().addAll(t, rx, ry, rz);
        }
    }
    //------------------------------------

    /*
    Input shape = Rank: 2, DataType: FLOAT, Offset: 0, Order: c, Shape: [32,200],  Stride: [200,1]
    features are [32,200]    layer sizes are 250, 150, 250, 250, 250. 200*250=50,000  150*250=37,500  250*250=62,500,
    layer size(i) are #biases+ LayerSize(i-1)*LayerSize(i).
    shape of features = Rank: 2, DataType: FLOAT, Offset: 0, Order: c, Shape: [32,200],  Stride: [200,1]
    params of layer 0   are: Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,50250],  Stride: [1,1]
       W => Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [200,250],  Stride: [1,200]
       b => Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,250],  Stride: [1,1]
    params of layer 1   are: Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,37650],  Stride: [1,1]
       W => Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [250,150],  Stride: [1,250]
       b => Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,150],  Stride: [1,1]
    params of layer 2   are: Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,37750],  Stride: [1,1]
       W => Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [150,250],  Stride: [1,150]
       b => Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,250],  Stride: [1,1]
    params of layer 3   are: Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,62750],  Stride: [1,1]
       W => Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [250,250],  Stride: [1,250]
       b => Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,250],  Stride: [1,1]
    params of layer 4   are: Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,62750],  Stride: [1,1]
       W => Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [250,250],  Stride: [1,250]
       b => Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,250],  Stride: [1,1]
    params of layer 5   are: Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,62750],  Stride: [1,1]
       W => Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [250,250],  Stride: [1,250]
       b => Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,250],  Stride: [1,1]
    params of layer 6   are: Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,50200],  Stride: [1,1]
       W => Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [250,200],  Stride: [1,250]
       b => Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,200],  Stride: [1,1]

       For LSTM:
0:[W, RW, b]
   Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [77,800],  Stride: [1,77]
   Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [200,800],  Stride: [1,200]
   Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,800],  Stride: [1,1]
1:[W, RW, b]
   Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [200,800],  Stride: [1,200]
   Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [200,800],  Stride: [1,200]
   Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,800],  Stride: [1,1]
2:[W, b]
   Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [200,77],  Stride: [1,200]
   Rank: 2, DataType: FLOAT, Offset: 0, Order: f, Shape: [1,77],  Stride: [1,1]

     */
    public void buildImages() {
        final int imageFactor=4;
        LocalTime localTime = LocalTime.now();
        setTitle("Weights as of " + LocalTime.now().format(dateTimeFormatter) + ", Click and drag to rotate. Arrow keys move.  Ctrl-click on image and drag to move image.");
        int imageIndex = 0;
        for(int i=0;i<net.getnLayers();i++) {
            //INDArray params = net.getLayer(i).params();
            //System.out.println("params of layer "  + i + "   are: " + params.shapeInfoToString());
            Map<String, INDArray> map = net.getLayer(i).paramTable();
            //INDArray biases = map.get("b");
            for(INDArray weights: map.values()) {
                if (weights.rank()!=2) {
                    continue;
                }
                final int rows = weights.rows();
                final int cols = weights.columns();
                if (rows==1 || cols == 1) {
                    continue; // TODO
                }
                WritableImage image = new WritableImage((1 + rows) * imageFactor, cols * imageFactor);
                PixelWriter pixelWriter = image.getPixelWriter();
                ImageView imageView = imageViews.get(imageIndex);
                imageIndex++;
                imageView.setImage(image);
//            imageView.setFitWidth(image.getWidth());
//            imageView.setFitHeight(image.getHeight());

                float max = Float.NEGATIVE_INFINITY;
                float min = Float.POSITIVE_INFINITY;
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        float f = weights.getFloat(row, col);
                        max = Math.max(f, max);
                        min = Math.min(f, min);
                    }
                }
                float range = Math.max(0.00000001f, max - min);
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        int argb = toArgb(min, max, range, weights.getFloat(row, col));
                        for (int rowOffset = 0; rowOffset < imageFactor; rowOffset++) {
                            for (int colOffset = 0; colOffset < imageFactor; colOffset++) {
                                pixelWriter.setArgb(row * imageFactor + rowOffset, col * imageFactor + colOffset, argb);
                            }
                        }
                    }
                }
            }
            //System.out.print("    " + biases.shapeInfoToString());
            //System.out.print("   " + weights.shapeInfoToString());
        }
    }
    private void setupImages() {
        final Point3D axis = new Point3D(1,0,0);
        int countOfImages = 0;
        for (int layerIndex = 0; layerIndex < net.getnLayers(); layerIndex++) {
            Map<String, INDArray> map = net.getLayer(layerIndex).paramTable();
            for (String key: map.keySet()) {
                INDArray weights = map.get(key);
                System.out.println("Layer + " + layerIndex + ", key = " + key + ": " + weights.shapeInfoToString());
                if (weights.rank()!=2) {
                    continue;
                }
                final int rows = weights.rows();
                final int cols = weights.columns();
                if (rows == 1 || cols == 1) {
                    continue; // TODO
                }
                ImageView imageView = new ImageView();
                imageView.setUserData(layerIndex + " " + key);
                world.getChildren().add(imageView);
                imageView.setTranslateZ(countOfImages * 200);
                imageView.setTranslateX(countOfImages * 20);
                imageView.setRotationAxis(axis);
               // imageView.setRotate(82);
                imageViews.add(imageView);
                Tooltip tooltip = new Tooltip("Layer " + layerIndex + ": " + key);
                Tooltip.install(imageView, tooltip);
                tooltip.setShowDelay(Duration.seconds(1));
                countOfImages++;
            }
        }
        System.out.println("Ys:");
        for(ImageView imageView:imageViews) {
            System.out.println(imageView.getUserData() + ": " + imageView.getTranslateY());
        }
    }
    private static int toArgb(float min, float max, float range, float f) {
        float proportion = (f-min)/range;
        return java.awt.Color.getHSBColor(proportion, 1.0f,1.0f).getRGB(); // TODO
    }
    private void handleMouse(Scene scene) {
        scene.setOnMousePressed((MouseEvent me) -> {
            mouseOldX = mousePosX = me.getSceneX();
            mouseOldY = mousePosY = me.getSceneY();
            Node node = me.getPickResult().getIntersectedNode();
            if (me.isControlDown() && node instanceof ImageView) {
                selectedImageView = (ImageView)node;
            }
        });
        scene.setOnMouseReleased((MouseEvent me) -> {
            selectedImageView = null;
        });
        scene.setOnMouseDragExited((MouseEvent me) -> {
        });
        scene.setOnMouseDragged((MouseEvent me) -> {
            mouseOldX = mousePosX;
            mouseOldY = mousePosY;
            mousePosX = me.getSceneX();
            mousePosY = me.getSceneY();

            mouseDeltaX = (mousePosX - mouseOldX);
            mouseDeltaY = (mousePosY - mouseOldY);
            if (selectedImageView == null) {
                if (me.isPrimaryButtonDown()) {
                    world.ry.setAngle(world.ry.getAngle() - mouseDeltaX * 0.2);
                    world.rx.setAngle(world.rx.getAngle() + mouseDeltaY * 0.2);
                } else if (me.isSecondaryButtonDown()) {
                    world.t.setZ(world.t.getZ() - mouseDeltaY);
                    world.t.setX(world.t.getX() + mouseDeltaX);
                }
            } else {
                selectedImageView.setTranslateX(selectedImageView.getTranslateX() - mouseDeltaX);
                selectedImageView.setTranslateY(selectedImageView.getTranslateY() + mouseDeltaY);
            }
        });
    }
    private void handleKeyEvent(KeyEvent keyEvent) {
        int delta = keyEvent.isShiftDown() ? 100: 10;
        switch (keyEvent.getCode()) {
            case LEFT:
                camera.setTranslateX(camera.getTranslateX() + delta);
                return;
            case RIGHT:
                camera.setTranslateX(camera.getTranslateX() - delta);
                return;
            case UP:
                if (keyEvent.isControlDown()) {
                    camera.setTranslateY(camera.getTranslateY() - delta);
                } else {
                    camera.setTranslateZ(camera.getTranslateZ() + delta);
                }
                return;
            case DOWN:
                if (keyEvent.isControlDown()) {
                    camera.setTranslateY(camera.getTranslateY() + delta);
                } else {
                    camera.setTranslateZ(camera.getTranslateZ() - delta);
                }
                return;
            default:
        }
        System.out.println(camera.getTranslateX() + ", " + camera.getTranslateY() + ", " + camera.getTranslateZ());
    }
    private void buildCamera() {
        root.getChildren().add(cameraXform);
        cameraXform.getChildren().add(camera);
        camera.setNearClip(CAMERA_NEAR_CLIP);
        camera.setFarClip(CAMERA_FAR_CLIP);
        camera.setTranslateX(cameraInitialX);
        camera.setTranslateY(cameraInitialY);
        camera.setTranslateZ(cameraInitialZ);
        camera.setRotationAxis(YAXIS);
    }
    public WeightsImageStage(MultiLayerNetwork net) {
        super(StageStyle.DECORATED);
        this.net = net;
        setTitle("Weights.  Click and drag to rotate. Arrow keys move.  Ctrl-click on image and drag to move image.");
        scene = new Scene(root, width, height, true);
        scene.setFill(new Color(0.0, 0.0, 0.0, 1));
        root.getChildren().add(world);
        world.setDepthTest(DepthTest.ENABLE);
        setScene(scene);
        handleMouse(scene);
        scene.setCamera(camera);
        buildCamera();
        scene.setOnKeyPressed(ke -> handleKeyEvent(ke));
        setupImages();
        show();
    }
}
