package projekt.xls;

import org.jxls.common.Context;
import org.jxls.util.JxlsHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import projekt.Group;

import java.io.*;
import java.util.Arrays;
import java.util.List;

public class XlsCreator {
    private static final Logger logger = LoggerFactory.getLogger(XlsCreator.class);

    public static void createXls(List<Group> groups, String fileName){
        logger.info("Running Object Collection demo");

        try(InputStream is = XlsCreator.class.getClassLoader().getResourceAsStream("template2.xls")) {
            try (OutputStream os = new FileOutputStream(fileName)) {
                Context context = new Context();
                context.putVar("groups", groups);
                context.putVar("sheetNames", Arrays.asList("Sheet 1", "Sheet 2", "Sheet 3", "Sheet 4"));
                //context.putVar("persons", displayPersons);
                JxlsHelper.getInstance().processTemplate(is, os, context);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }


    }
}
