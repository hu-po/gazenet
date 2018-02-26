using UnityEngine;

public class RandomPose
{
    // Local position and rotation
    public float[] pos, rot;

    // transform of object
    public Transform t;

    // Translation and Rotation constraints
    public float[] xrange, yrange, zrange, rxrange, ryrange, rzrange;

    public RandomPose(Transform objTransform)
    {
        t = objTransform;
        // Default local pose is zero
        pos = new float[3] { 0.0f, 0.0f, 0.0f };
        rot = new float[3] { 0.0f, 0.0f, 0.0f };

        // Default values for ranges are all centered on zero
        xrange = new float[2] { 0.0f, 0.0f };
        yrange = new float[2] { 0.0f, 0.0f };
        zrange = new float[2] { 0.0f, 0.0f };
        rxrange = new float[2] { 0.0f, 0.0f };
        ryrange = new float[2] { 0.0f, 0.0f };
        rzrange = new float[2] { 0.0f, 0.0f };
    }

    public void nextPose()
    {
        // New location is picked randomly from constraints
        pos[0] = Random.Range(xrange[0], xrange[1]); // X
        pos[1] = Random.Range(yrange[0], yrange[1]); // Y
        pos[2] = Random.Range(zrange[0], zrange[1]); // Z
        rot[0] = Random.Range(rxrange[0], rxrange[1]); // RX
        rot[1] = Random.Range(ryrange[0], ryrange[1]); // RY
        rot[2] = Random.Range(rzrange[0], rzrange[1]);// RZ
        // Set the local pose of the transform
        t.localPosition = new Vector3(pos[0], pos[1], pos[2]);
        t.localEulerAngles = new Vector3(rot[0], rot[1], rot[2]);
    }
}
