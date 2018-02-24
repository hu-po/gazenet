using System.IO;
using UnityEngine;

public class GazeCamera : MonoBehaviour {

    private Camera cam;

    void Start()
    {
        // Assign local camera object
        cam = this.gameObject.GetComponent<Camera>()

    }

    private void SaveImage(int cullMask, string dataPath, string imageName)
    {
        this.
        // Renders camera to a texture, saves texture to a PNG image
        this.gameObject.camera.Render();
        RenderTexture.active = gazeCamera.targetTexture;
        Texture2D image = new Texture2D(gazeCamera.targetTexture.width, gazeCamera.targetTexture.height, TextureFormat.ARGB32, false);
        image.ReadPixels(new Rect(0, 0, gazeCamera.targetTexture.width, gazeCamera.targetTexture.height), 0, 0);
        image.Apply();
        RenderTexture.active = null;
        byte[] bytes = image.EncodeToPNG();
        File.WriteAllBytes(Path.Combine(dataPath, imageName), bytes);
    }
}
