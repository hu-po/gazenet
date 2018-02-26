using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LightController : MonoBehaviour {

    // Individual scene lights
    public GameObject frontLight;
    public GameObject topLight;

    // Pose randomizers for each of the lights
    private RandomPose frontLightPose;
    private RandomPose topLightPose;

	// Use this for initialization
	void Start () {

        // Define the extent of box for random body placement
        frontLightPose = new RandomPose(frontLight.transform)
        {
            xrange = new float[] { -0.25f, 0.25f },
            yrange = new float[] { -0.25f, 0.25f },
            zrange = new float[] { 0.1f, 0.8f }
        };

        // Define the extent of random pose for the look point
        topLightPose = new RandomPose(topLight.transform)
        {
            xrange = new float[] { -1.0f, 1.0f },
            yrange = new float[] { 0.2f, 0.8f },
            zrange = new float[] { -0.05f, 0.15f }
        };

    }
	
	public void StepLights()
    {
        // Get a new random pose for both lights
        frontLightPose.nextPose();
        topLightPose.nextPose();

        // Vary the hue, intensity
        VaryLight(frontLight.GetComponent<Light>());
        VaryLight(topLight.GetComponent<Light>());
    }

    private void VaryLight(Light light)
    {
        light.intensity = Random.Range(0.5f, 1.0f);
    }
}
