using System.IO;
using UnityEngine;

public class Dataset : MonoBehaviour {


    // Controller scripts provide utilities
    private GazemanController man;
    private CameraController cam;
    private LightController lights;

    // Directories for data storage
    private string dataPath;
    private string imageName;

	// Use this for initialization
	void Start ()
    {
        // Dataset name is based on current date and time
        dataPath = Path.Combine(Path.Combine(Application.dataPath, "../../../data"), System.DateTime.Now.ToString("ddMMyyyy_hhmm"));
        if (Directory.Exists(dataPath))
        {
            // Empty directory if it already exists (true deletes recursively)
            Directory.Delete(dataPath, true);
        }
        // Create directory to save images inside
        Directory.CreateDirectory(dataPath);

        // Get local references to controller scripts
        cam = gameObject.GetComponent<CameraController>();
        man = gameObject.GetComponent<GazemanController>();
        lights = gameObject.GetComponent<LightController>();

        // Start data collection
        float startTime = 2.0f; // Start time in seconds
        float delay = 0.02f; // Delay needed to make sure camera acptures updated scene
        float repeatTime = 0.04f; // Repeat time in seconds
        InvokeRepeating("UpdateScene", startTime, repeatTime);
        InvokeRepeating("SaveImage", startTime + delay, repeatTime);
    }

    private void UpdateScene()
    {
        // Randomize Lights
        lights.StepLights();
        // Move head within head space and look point within screen
        float[] lookPos = man.StepGaze();
        // Add 0.5 to remove the negative values
        lookPos[0] += 0.5f;
        lookPos[1] += 0.5f;
        imageName = lookPos[0].ToString("0.000") + '_' + lookPos[1].ToString("0.000") + ".png";
    }

    private void SaveImage()
    {
        // Save segmentation image
        cam.Snapshot(dataPath, imageName);
    }
}
