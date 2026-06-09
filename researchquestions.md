```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#ffffff' }}}%%

graph LR
    %% Main Root
    Root((Pure Mobility))
    
    %% Level 1: Project break down
    Root --> Vehicle[<b>Vehicle Concepts</b>]
    Root --> Fleet[<b>Fleet Management</b>]
    Root --> Road[<b>Road Concepts</b>]

    %% Level 2: Vehicle Concepts Breakdown
    Vehicle --> V_Pax[<b>Passenger Vehicles</b>]
    Vehicle --> V_Car[<b>Cargo Vehicles</b>]
    Vehicle --> V_Co[<b>Co-Modal Vehicles</b>]

    %% Level 2: Fleet Management Breakdown
    Fleet --> F_Dem[<b>Demand</b>]
    Fleet --> F_Pl[<b>Planing</b><br/>Algorithms (dispatching, routing, scheduling)]
    Fleet --> F_Op[<b>Operation</b><br/>real-time execution, handling disruptions]
    Fleet --> F_Con[<b>Context</b><br/>legal, infrastructure, interfaces]

    %% Level 3: Demand breakdown
    F_Dem --> F_Dem_Pax[<b>Passenger</b>]
    F_Dem --> F_Dem_Cargo[<b>Cargo</b>]
    
    %% Level 3: Context Breakdown
    F_Con --> F_Con_Legal[<b>Legal</b>]
    F_Con --> F_Con_Infr[<b>Infrastructure</b>]
    F_Con --> F_Con_Inter[<b>Interfaces</b><br/>data, material, humans]


    %% Styling for visual clarity
    classDef main fill:#f9f,stroke:#333,stroke-width:4px;
    classDef pillar fill:#bbf,stroke:#333,stroke-width:2px;
    classDef sub fill:#dfd,stroke:#333,stroke-width:1px;
    classDef leaf fill:#fff,stroke:#333,stroke-width:1px,text-align:left;

    class Root main;
    class Vehicle,AMoD,Road pillar;
    class A_In,A_Op,A_Out,A_Con sub;
    class A_In_Pax,A_In_Cargo,A_Op_Global,A_Op_Local,A_Con_Legal,A_Con_Infr,A_Con_Inter leaf; 

    %% Arrow (link) styling
    %% NOTE: Since background is #ffffff, 'white' arrows will be invisible. 
    %% Change 'white' to another color like 'black' or '#ff0000' if needed.
    linkStyle default stroke:white,stroke-width:2px;

```

