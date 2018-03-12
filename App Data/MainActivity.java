package blueguy.decentlogger;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;

public class MainActivity extends Activity {
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        finish();
        startService(new Intent(this, LoggingService.class));
    }
}
