package blueguy.decentlogger;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener2;
import android.util.Log;
import java.io.IOException;
import java.util.zip.GZIPOutputStream;

final class C0116b implements SensorEventListener2 {
    final /* synthetic */ GZIPOutputStream f595a;

    C0116b(GZIPOutputStream gZIPOutputStream) {
        this.f595a = gZIPOutputStream;
    }

    public final void onAccuracyChanged(Sensor sensor, int i) {
    }

    public final void onFlushCompleted(Sensor sensor) {
        try {
            Log.d("Decent_LoggingService", "Flushing & Closing Stream");
            this.f595a.finish();
            this.f595a.close();
        } catch (IOException e) {
            Log.e("Decent_LoggingService", "failed flush");
        }
    }

    public final void onSensorChanged(SensorEvent sensorEvent) {
        String str = ",";
        for (float f : sensorEvent.values) {
            str = str + f + ",";
        }
        String str2 = (LoggingService.f584b + (sensorEvent.timestamp / 1000000)) + str + sensorEvent.accuracy + "," + LoggingService.f588f + "\n";
        try {
            this.f595a.write(str2.getBytes());
        } catch (IOException e) {
            Log.e("Decent_LoggingService", "failed: " + str2);
        }
    }
}
