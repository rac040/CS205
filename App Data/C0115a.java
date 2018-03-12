package blueguy.decentlogger;

import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.hardware.TriggerEvent;
import android.hardware.TriggerEventListener;
import android.util.Log;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.zip.GZIPOutputStream;

final class C0115a extends TriggerEventListener {
    final /* synthetic */ File f592a;
    final /* synthetic */ Sensor f593b;
    final /* synthetic */ SensorManager f594c;

    C0115a(File file, Sensor sensor, SensorManager sensorManager) {
        this.f592a = file;
        this.f593b = sensor;
        this.f594c = sensorManager;
    }

    public final void onTrigger(TriggerEvent triggerEvent) {
        if (LoggingService.f585c) {
            String str = ",";
            for (float f : triggerEvent.values) {
                str = str + f + ",";
            }
            String str2 = (LoggingService.f584b + (triggerEvent.timestamp / 1000000)) + str + LoggingService.f588f + "\n";
            try {
                GZIPOutputStream gZIPOutputStream = new GZIPOutputStream(new FileOutputStream(new File(this.f592a, this.f593b.getType() + "_" + this.f593b.getStringType() + ".data.csv.gz"), true));
                gZIPOutputStream.write(str2.getBytes());
                gZIPOutputStream.finish();
                gZIPOutputStream.close();
            } catch (IOException e) {
                Log.e("Decent_LoggingService", "failed: " + str2);
            }
            this.f594c.requestTriggerSensor(LoggingService.m728b(this.f592a, this.f593b, this.f594c), this.f593b);
        }
    }
}
