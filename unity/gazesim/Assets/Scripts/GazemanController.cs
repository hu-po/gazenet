using UnityEngine;

public class GazemanController : MonoBehaviour {


    // This should be drag-assigned in Editor
    // The look point is where the eyes will stare at
    public GameObject lookPoint;
    public GameObject head;
    public GameObject eyeL;
    public GameObject eyeR;
    public GameObject body;

    // Extent of random location for body
    private RandomPose gazemanPose;
    private RandomPose lookpointPose;


    private void Start()
    {

        // Define the extent of box for random body placement
        gazemanPose = new RandomPose(body.transform)
        {
            xrange = new float[] { -0.17f, 0.17f },
            yrange = new float[] { -0.06f, 0.06f },
            zrange = new float[] { -0.05f, 0.15f },
            rxrange = new float[] { -10.0f, 6.0f }
        };

        // Define the extent of random pose for the look point
        lookpointPose = new RandomPose(lookPoint.transform)
        {
            xrange = new float[] { -0.5f, 0.5f },
            yrange = new float[] { -0.5f, 0.5f }
        };
    }

    public float[] StepGaze()
    {
        // Get a new random pose for the body and the lookpoint
        gazemanPose.nextPose();
        lookpointPose.nextPose();

        // Use LookAt function to get head to look at lookpoint
        head.transform.LookAt(lookPoint.transform.position);

        // Return the position array for the look pos
        return lookpointPose.pos;
    }
}
