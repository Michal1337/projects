package projekt;

import java.util.List;

public class Group {
    private String name;
    private List<DisplayPerson> displayPersonList;

    public Group(String name, List<DisplayPerson> displayPersonList) {
        this.name = name;
        this.displayPersonList = displayPersonList;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<DisplayPerson> getDisplayPersonList() {
        return displayPersonList;
    }

    public void setDisplayPersonList(List<DisplayPerson> displayPersonList) {
        this.displayPersonList = displayPersonList;
    }
}
