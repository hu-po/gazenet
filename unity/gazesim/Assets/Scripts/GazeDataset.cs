using System.IO;
using UnityEngine;

public class GazeDataset : MonoBehaviour {
    
    // Head contains the left and right eyes
    public GameObject head;

    // The look point is where the eyes will stare at
    public GameObject lookPoint;

    // Camera used for gaze tracking
    public Camera gazeCamera;

    // Directories for data storage
    private string datasetName = "100118_fixedhead";
    private string dataPath;
    private string imageName;

	// Use this for initialization
	void Start () {

        // Create directory to save images inside
        SetupDataDir();

        // Start data collection
        float startTime = 1.0f; // Start time in seconds
        float delay = 0.02f; // Delay needed to make sure camera acptures updated scene
        float repeatTime = 0.04f; // Repeat time in seconds
        InvokeRepeating("UpdateGazeScene", startTime, repeatTime);
        InvokeRepeating("SaveImage", startTime + delay, repeatTime);
    }

    private void UpdateGazeScene()
    {
        // Move head within head space and look point within screen
        float[] lookPos = MoveLookPoint();
        //float[] headPos = MoveHead();

        // Make the eyes look at the look point
        RotateHead();

        // Name of saved image contains the target (lookPos)
        lookPos[0] += 0.5f; // Add 0.5 to remove negative values
        lookPos[1] += 0.5f;
        imageName = lookPos[0].ToString("0.00") + '_' + lookPos[1].ToString("0.00") + ".png";
    }

    private void SetupDataDir()
    {
        dataPath = Path.Combine(Path.Combine(Application.dataPath, "../Data"), datasetName);
        if (Directory.Exists(dataPath))
        {
            // Empty directory if it already exists (true deletes recursively)
            Directory.Delete(dataPath, true);
        }
        Directory.CreateDirectory(dataPath);
    }

    private float[] MoveLookPoint()
    {
        // Choose new random location for look point
        float[] newPos = new float[2];
        newPos[0] = Random.Range(-0.5f, 0.5f); // X
        newPos[1] = Random.Range(-0.5f, 0.5f); // Y
        // Move the lookpoint within the screen by changing local position
        lookPoint.transform.localPosition = new Vector3(newPos[0], newPos[1], 0);
        return newPos;
    }

    private float[] MoveHead()
    {
        // Choose new location for the head
        float[] newPos = new float[3];
        newPos[0] = Random.Range(-0.5f, 0.5f); // X
        newPos[1] = Random.Range(-0.5f, 0.5f); // Y
        newPos[2] = Random.Range(-0.5f, 0.5f); // Z
        // Move the head within the head space by changing local position
        head.transform.localPosition = new Vector3(newPos[0], newPos[1], newPos[2]);
        return newPos;
    }

    private void RotateHead()
    {
        // Use LookAt function to get head to look at lookpoint
        head.transform.LookAt(lookPoint.transform.position);

        // TODO: More complicated motion with head and eyes being seperate
    }

    private void SaveImage()
    {
        // Renders camera to a texture, saves texture to a PNG image
        gazeCamera.Render();
        RenderTexture.active = gazeCamera.targetTexture;
        Texture2D image = new Texture2D(gazeCamera.targetTexture.width, gazeCamera.targetTexture.height, TextureFormat.ARGB32, false);
        image.ReadPixels(new Rect(0, 0, gazeCamera.targetTexture.width, gazeCamera.targetTexture.height), 0, 0);
        image.Apply();
        RenderTexture.active = null;
        byte[] bytes = image.EncodeToPNG();
        File.WriteAllBytes(Path.Combine(dataPath, imageName), bytes);
    }
}
