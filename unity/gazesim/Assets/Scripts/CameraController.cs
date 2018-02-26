using System.IO;
using UnityEngine;

public class CameraController : MonoBehaviour {

    public Camera cam;

    // Attributes to be changed by external scripts
    public int imageWidth;
    public int imageHeight;

	void Start () {

        // Target texture for camera should match image width and height
        cam.aspect = (float)imageWidth / (float)imageHeight;
        cam.targetTexture.width = imageWidth;
        cam.targetTexture.height = imageHeight;

    }

    public void Snapshot(string dataPath, string imageName, bool segmented = false)
    {
        // Save the "Normal" image
        string savePath = Path.Combine(dataPath, imageName);
        SaveImage(savePath);

        // Save the Segmentation image
        if (segmented)
        { 
            ToggleMask("Background");
            // Seperate folder for segmented images
            string segPath = Path.Combine(dataPath, "seg");
            Directory.CreateDirectory(segPath);
            savePath = Path.Combine(segPath, imageName);
            SaveImage(savePath);
            ToggleMask("Background");
        }
    }

    private void ToggleMask(string mask)
    {
        // Toggle the bit using a XOR operation
        cam.cullingMask ^= 1 << LayerMask.NameToLayer(mask);
    }

    private void SaveImage(string savePath)
    {
        // Renders camera to a texture, saves texture to a PNG image
        cam.Render();
        RenderTexture.active = cam.targetTexture;
        Texture2D image = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        image.Apply();
        RenderTexture.active = null;
        byte[] bytes = image.EncodeToPNG();
        File.WriteAllBytes(savePath, bytes);
    }

}
