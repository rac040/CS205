package blueguy.decentlogger;

import android.app.Notification;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.res.Resources;
import android.hardware.Sensor;
import android.hardware.SensorEventListener;
import android.hardware.SensorEventListener2;
import android.hardware.SensorManager;
import android.hardware.TriggerEventListener;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.IBinder;
import android.os.PowerManager;
import android.os.PowerManager.WakeLock;
import android.os.SystemClock;
import android.support.v4.app.aw;
import android.support.v4.app.ay;
import android.support.v4.app.bb;
import android.support.v4.app.bd;
import android.support.v4.app.br;
import android.support.v4.app.ch;
import android.support.v4.app.cj;
import android.util.Log;
import android.widget.Toast;
import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.zip.GZIPOutputStream;

public class LoggingService extends Service {
    private static WakeLock f583a = null;
    private static long f584b = (System.currentTimeMillis() - SystemClock.elapsedRealtime());
    private static boolean f585c = false;
    private static List f586d = new ArrayList();
    private static List f587e = new ArrayList();
    private static String f588f = "unknown";
    private static final String f589g = Environment.getExternalStorageDirectory().getAbsolutePath();
    private static final SimpleDateFormat f590h = new SimpleDateFormat("'_'EEE_MMM_dd_HH-mm_yyyy_zzz", Locale.US);
    private static final String f591i = (f589g + "/Sessions/");

    private static void m723a(Context context, boolean z) {
        if (z) {
            m729b(context, true);
            m725a(context);
            Toast.makeText(context, "started", 0).show();
            return;
        }
        SensorManager sensorManager = (SensorManager) context.getSystemService("sensor");
        f585c = false;
        for (SensorEventListener2 sensorEventListener2 : f586d) {
            try {
                sensorManager.flush(sensorEventListener2);
                sensorManager.unregisterListener(sensorEventListener2);
                sensorEventListener2.onFlushCompleted(sensorManager.getDefaultSensor(1));
            } catch (Exception e) {
                Log.e("Decent_LoggingService", "Stopping listener " + sensorEventListener2.toString() + " failed!");
            }
        }
        f586d = new ArrayList();
        m729b(context, false);
        Toast.makeText(context, "stopped", 0).show();
    }

    private static boolean m725a(Context context) {
        SensorManager sensorManager = (SensorManager) context.getSystemService("sensor");
        List<Sensor> sensorList = sensorManager.getSensorList(-1);
        Log.i("Decent_LoggingService", "Total Sensor Count : " + sensorList.size());
        if (sensorList.isEmpty()) {
            return false;
        }
        String str = Build.SERIAL + f590h.format(new Date());
        File file = new File(f591i + str + "/attr/");
        File file2 = new File(f591i + str + "/data/");
        file.mkdirs();
        file2.mkdirs();
        if (file.canWrite() && file2.canWrite()) {
            f584b = System.currentTimeMillis() - SystemClock.elapsedRealtime();
            for (Sensor sensor : sensorList) {
                try {
                    String str2;
                    int maxDelay;
                    BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(file, sensor.getType() + "_" + sensor.getStringType() + ".attr.csv")));
                    switch (sensor.getReportingMode()) {
                        case 0:
                            str2 = "continuous";
                            maxDelay = sensor.getMaxDelay();
                            break;
                        case 1:
                            str2 = "on change";
                            maxDelay = sensor.getMaxDelay();
                            break;
                        case 2:
                            str2 = "one shot";
                            maxDelay = -1;
                            break;
                        case 3:
                            str2 = "special trigger";
                            maxDelay = -1;
                            break;
                        default:
                            str2 = String.valueOf(sensor.getReportingMode());
                            maxDelay = -1;
                            break;
                    }
                    bufferedWriter.write("type," + sensor.getType() + "\n");
                    bufferedWriter.write("string type," + sensor.getStringType() + "\n");
                    bufferedWriter.write("name," + sensor.getName() + "\n");
                    bufferedWriter.write("reporting made," + str2 + "\n");
                    bufferedWriter.write("wake up," + sensor.isWakeUpSensor() + "\n");
                    bufferedWriter.write("max delay (μs)," + maxDelay + "\n");
                    bufferedWriter.write("min delay (μs)," + sensor.getMinDelay() + "\n");
                    bufferedWriter.write("fifo max event count," + sensor.getFifoMaxEventCount() + "\n");
                    bufferedWriter.write("fifo reserved event count," + sensor.getFifoReservedEventCount() + "\n");
                    bufferedWriter.write("max range," + sensor.getMaximumRange() + "\n");
                    bufferedWriter.write("max resolution," + sensor.getResolution() + "\n");
                    bufferedWriter.write("power (mA)," + sensor.getPower() + "\n");
                    bufferedWriter.write("version," + sensor.getVersion() + "\n");
                    bufferedWriter.write("vendor," + sensor.getVendor() + "\n");
                    bufferedWriter.flush();
                    bufferedWriter.close();
                    int minDelay = sensor.getMinDelay() > 0 ? sensor.getMinDelay() : 1;
                    maxDelay = sensor.getMaxDelay() * sensor.getFifoMaxEventCount();
                    switch (sensor.getReportingMode()) {
                        case 0:
                        case 1:
                            GZIPOutputStream gZIPOutputStream = new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(new File(file2, sensor.getType() + "_" + sensor.getStringType() + ".data.csv.gz"))));
                            Log.d("Decent_LoggingService", "Opening Continuous Stream " + sensor.getType());
                            SensorEventListener c0116b = new C0116b(gZIPOutputStream);
                            f586d.add(c0116b);
                            sensorManager.registerListener(c0116b, sensor, minDelay, maxDelay);
                            break;
                        case 2:
                            f585c = true;
                            sensorManager.requestTriggerSensor(m728b(file2, sensor, sensorManager), sensor);
                            break;
                        default:
                            Log.e("Decent_LoggingService", "Dunno how to start " + sensor.getStringType());
                            break;
                    }
                } catch (Exception e) {
                    Log.e("Decent_LoggingService", "Starting sensor " + sensor.getStringType() + " failed!");
                }
            }
            return true;
        }
        Log.e("Decent_LoggingService", "Directory not writable. Exiting");
        return false;
    }

    private static Notification m727b(Context context) {
        boolean z = !f588f.equals("unknown");
        long currentTimeMillis = System.currentTimeMillis();
        Resources resources = context.getResources();
        cj cjVar = new cj("extra_voice_reply");
        cjVar.f230c = resources.getStringArray(R.array.reply_choices);
        cjVar.f229b = "Positioning Label?";
        cjVar.f231d = true;
        ch chVar = new ch(cjVar.f228a, cjVar.f229b, cjVar.f230c, cjVar.f231d, cjVar.f232e);
        ay ayVar = new ay(z ? "Update Label" : "Start Session", PendingIntent.getService(context, 0, new Intent(context, LoggingService.class), 134217728));
        if (ayVar.f121f == null) {
            ayVar.f121f = new ArrayList();
        }
        ayVar.f121f.add(chVar);
        aw awVar = new aw(ayVar.f116a, ayVar.f117b, ayVar.f118c, ayVar.f120e, ayVar.f121f != null ? (ch[]) ayVar.f121f.toArray(new ch[ayVar.f121f.size()]) : null, ayVar.f119d);
        bd brVar = new br();
        brVar.m115a(64, false);
        brVar.m115a(1, true);
        brVar.f174b = 80;
        brVar.f173a = R.mipmap.ic_launcher;
        brVar.m115a(2, !z);
        bb bbVar = new bb(context);
        bbVar.f147j = 2;
        bbVar = bbVar.m91a((int) R.mipmap.ic_launcher);
        bbVar.f149l = z;
        bbVar = bbVar.m93a(f588f);
        brVar.mo17a(bbVar);
        bbVar = bbVar.m94b("--:--");
        bbVar.f159v.add(awVar);
        return bbVar.m92a(currentTimeMillis).m90a();
    }

    private static TriggerEventListener m728b(File file, Sensor sensor, SensorManager sensorManager) {
        Log.d("Decent_LoggingService", "Opening Trigger Stream " + sensor.getType());
        return new C0115a(file, sensor, sensorManager);
    }

    private static boolean m729b(Context context, boolean z) {
        PowerManager powerManager = (PowerManager) context.getSystemService("power");
        if (f583a == null) {
            f583a = powerManager.newWakeLock(1, "Decent_LoggingService");
        }
        boolean isHeld = f583a.isHeld();
        if (isHeld && !z) {
            f583a.release();
        }
        if (!isHeld && z) {
            f583a.acquire();
        }
        return f583a.isHeld();
    }

    public IBinder onBind(Intent intent) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    public void onCreate() {
        startForeground(372, m727b(this));
    }

    public void onDestroy() {
        m723a(this, false);
        stopForeground(true);
    }

    public int onStartCommand(Intent intent, int i, int i2) {
        CharSequence charSequence;
        if (intent == null) {
            charSequence = "null";
        } else {
            Bundle a = ch.m148a(intent);
            if (a == null) {
                charSequence = "null";
            } else {
                charSequence = a.getCharSequence("extra_voice_reply");
                if (charSequence == null) {
                    charSequence = "null";
                }
            }
        }
        String toLowerCase = charSequence.toString().replaceAll(" ", "_").toLowerCase();
        boolean equals = f588f.equals("unknown");
        boolean z = equals && toLowerCase.equals("null");
        boolean contains = toLowerCase.contains("unknown");
        equals = !contains && equals;
        if (!z) {
            f588f = toLowerCase;
            if (contains) {
                m723a(this, false);
            }
            if (equals) {
                m723a(this, true);
            }
            ((NotificationManager) getSystemService("notification")).notify(372, m727b(this));
        }
        return 1;
    }
}
