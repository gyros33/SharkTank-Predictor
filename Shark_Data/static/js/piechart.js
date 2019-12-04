var data = [{
    type: "sunburst",
    labels: [
        "Food",
        "Alcoholic Beverages", "Deal Made", "No Deal Made",
        "Non-Alcoholic Beverages", "Deal Made", "No Deal Made",
        "Specialty Food", "Deal Made", "No Deal Made",
        "Water Bottles", "Deal Made", "No Deal Made",
        "Wine Accessories", "Deal Made", "No Deal Made",
        "Other",
        "Automotive", "Deal Made", "No Deal Made",
        "Consumer Services", "Deal Made", "No Deal Made",
        "Costumes", "Deal Made", "No Deal Made",
        "Education", "Deal Made", "No Deal Made",
        "Novelties", "Deal Made", "No Deal Made",
        "Pet Products", "Deal Made", "No Deal Made",
        "Productivity Tools", "Deal Made", "No Deal Made",
        "Professional Services", "Deal Made", "No Deal Made",
        "Babies and Children", 
        "Baby and Child Care", "Deal Made", "No Deal Made", 
        "Baby and Children's Apparel and Accessories", "Deal Made", "No Deal Made", 
        "Baby and Children's Bedding", "No Deal Made", 
        "Baby and Children's Entertainment", "Deal Made", "No Deal Made",
        "Baby and Children's Food", "Deal Made", "No Deal Made", 
        "Health and Recreation", 
        "Cycling", "Deal Made",
        "Fitness Apparel and Accessories", "Deal Made", "No Deal Made",
        "Fitness Equipment", "Deal Made", "No Deal Made",
        "Fitness Programs", "Deal Made", "No Deal Made",
        "Gardening", "Deal Made", "No Deal Made",
        "Golf Products", "Deal Made", "No Deal Made",
        "Health and Well-Being", "Deal Made", "No Deal Made",
        "Homeopathic Remedies", "Deal Made", "No Deal Made",
        "Music", "Deal Made", "No Deal Made",
        "Outdoor Recreation", "Deal Made", "No Deal Made",
        "Personal Care and Cosmetics", "Deal Made", "No Deal Made",
        "Toys and Games", "Deal Made", "No Deal Made",
        "Technology", 
        "Electronics", "Deal Made", "No Deal Made", 
        "Entertainment", "Deal Made", "No Deal Made",
        "Mobile Apps", "Deal Made", "No Deal Made",
        "Online Services", "Deal Made", "No Deal Made",
        "Men and Women", 
        "Fashion Accessories", "No Deal Made",
        "Maternity", "No Deal Made",
        "Men and Women's Accessories", "Deal Made", "No Deal Made",
        "Men and Women's Apparel", "Deal Made", "No Deal Made",
        "Men and Women's Shoes", "Deal Made", "No Deal Made",
        "Men's Accessories", "Deal Made", "No Deal Made",
        "Undergarments and Basics", "Deal Made", "No Deal Made",
        "Weddings", "Deal Made", "No Deal Made",
        "Women's Accessories", "Deal Made", "No Deal Made",
        "Women's Apparel", "Deal Made", "No Deal Made",
        "Women's Shoes", "Deal Made", "No Deal Made",
        "Home", 
        "Furniture", "Deal Made", "No Deal Made", 
        "Holiday Cheer", "Deal Made", "No Deal Made",
        "Home Accessories", "Deal Made", "No Deal Made", 
        "Home Improvement", "Deal Made", 
        "Home Security Solutions", "Deal Made", "No Deal Made",
        "Kitchen Tools", "Deal Made", "No Deal Made",
        "Party Supplies", "Deal Made", "No Deal Made",
        "Pest Control", "Deal Made", "No Deal Made",
        "Storage and Cleaning Products", "Deal Made", "No Deal Made"],
    parents: [
        "",
        "Food", "Alcoholic Beverages", "Alcoholic Beverages",
        "Food", "Non-Alcoholic Beverages", "Non-Alcoholic Beverages",
        "Food", "Specialty Food", "Specialty Food", 
        "Food", "Water Bottles", "Water Bottles",
        "Food", "Wine Accessories", "Wine Accessories",
        "",
        "Other", "Automotive", "Automotive", 
        "Other", "Consumer Services", "Consumer Services",
        "Other", "Costumes", "Costumes",
        "Other", "Education", "Education",
        "Other", "Novelties", "Novelties",
        "Other", "Pet Products", "Pet Products",
        "Other", "Productivity Tools", "Productivity Tools",
        "Other", "Professional Services", "Professional Services",
        "",
        "Babies and Children", "Baby and Child Care", "Baby and Child Care",
        "Babies and Children", "Baby and Children's Apparel and Accessories", "Baby and Children's Apparel and Accessories",
        "Babies and Children", "Baby and Children's Bedding",
        "Babies and Children", "Baby and Children's Entertainment", "Baby and Children's Entertainment",
        "Babies and Children", "Baby and Children's Food", "Baby and Children's Food",
        "",
        "Health and Recreation", "Cycling",
        "Health and Recreation", "Fitness Apparel and Accessories", "Fitness Apparel and Accessories",
        "Health and Recreation", "Fitness Equipment", "Fitness Equipment",
        "Health and Recreation", "Fitness Programs", "Fitness Programs",
        "Health and Recreation", "Gardening", "Gardening",
        "Health and Recreation", "Golf Products", "Golf Products",
        "Health and Recreation", "Health and Well-Being", "Health and Well-Being",
        "Health and Recreation", "Homeopathic Remedies", "Homeopathic Remedies",
        "Health and Recreation", "Music", "Music",
        "Health and Recreation", "Outdoor Recreation", "Outdoor Recreation",
        "Health and Recreation", "Personal Care and Cosmetics", "Personal Care and Cosmetics",
        "Health and Recreation", "Toys and Games", "Toys and Games",
        "",
        "Technology", "Electronics", "Electronics",
        "Technology", "Entertainment", "Entertainment",
        "Technology", "Mobile Apps", "Mobile Apps",
        "Technology", "Online Services", "Online Services",
        "",
        "Men and Women", "Fashion Accessories",
        "Men and Women", "Maternity",
        "Men and Women", "Men and Women's Accessories", "Men and Women's Accessories",
        "Men and Women", "Men and Women's Apparel", "Men and Women's Apparel",
        "Men and Women", "Men and Women's Shoes", "Men and Women's Shoes",
        "Men and Women", "Men's Accessories", "Men's Accessories",
        "Men and Women", "Undergarments and Basics", "Undergarments and Basics",
        "Men and Women", "Weddings", "Weddings",
        "Men and Women", "Women's Accessories", "Women's Accessories",
        "Men and Women", "Women's Apparel", "Women's Apparel",
        "Men and Women", "Women's Shoes", "Women's Shoes",
        "",
        "Home", "Furniture", "Furniture",
        "Home", "Holiday Cheer", "Holiday Cheer",
        "Home", "Home Accessories", "Home Accessories",
        "Home", "Home Improvement",
        "Home", "Home Security Solutions", "Home Security Solutions",
        "Home", "Kitchen Tools", "Kitchen Tools",
        "Home", "Party Supplies", "Party Supplies",
        "Home", "Pest Control", "Pest Control",
        "Home", "Storage and Cleaning Products", "Storage and Cleaning Products"
    ],
    values: [
        500.0, 
        100.0, 75.0, 25.0,
        100.0, 60.0, 40.0,
        100.0, 54.1, 45.9,
        100.0, 66.7, 33.3,
        100.0, 33.3, 66.7,
        800.0,
        100.0, 60.0, 40.0,
        100.0, 23.1, 76.9,
        100.0, 50.0, 50.0,
        100.0, 75.0, 25.0,
        100.0, 45.7, 54.3,
        100.0, 46.2, 53.8,
        100.0, 40.0, 60.0,
        100.0, 20.0, 80.0,
        500.0,
        100.0, 66.7, 33.3,
        100.0, 87.5, 12.5,
        100.0, 100.0,
        100.0, 44.4, 55.6,
        100.0, 50.0, 50.0,
        1200.0,
        100.0, 100.0,
        100.0, 66.7, 33.3,
        100.0, 25.0, 75.0,
        100.0, 42.9, 57.1,
        100.0, 60.0, 40.0,
        100.0, 66.7, 33.3,
        100.0, 80.0, 20.0,
        100.0, 33.3, 66.7,
        100.0, 60.0, 40.0,
        100.0, 62.5, 37.5,
        100.0, 40.0, 60.0,
        100.0, 52.6, 47.4,
        400.0,
        100.0, 42.9, 57.1,
        100.0, 30.8, 69.2,
        100.0, 50.0, 50.0,
        100.0, 45.5, 54.5,
        1100.0,
        100.0, 100.0,
        100.0, 100.0,
        100.0, 25.0, 75.0,
        100.0, 33.3, 66.7,
        100.0, 20.0, 80.0,
        100.0, 40.0, 60.0,
        100.0, 28.6, 71.4,
        100.0, 33.3, 66.7,
        100.0, 62.5, 37.5,
        100.0, 50.0, 50.0,
        100.0, 75.0, 25.0,
        900.0,
        100.0, 60.0, 40.0,
        100.0, 75.0, 25.0,
        100.0, 57.1, 42.9,
        100.0, 100.0,
        100.0, 33.3, 66.7,
        100.0, 66.7, 33.3,
        100.0, 60.0, 40.0,
        100.0, 33.3, 66.7,
        100.0, 82.4, 17.6
    ],
    leaf: {"opacity": 0.4},
    marker: {line: {width: 2}},
    branchvalues:'total'
}];
var layout = {
    "margin": {"l": 0, "r": 0, "b": 0, "t": 0},
  };
Plotly.newPlot('myDiv', data, layout, {showSendToCloud: true})
myPlot = document.getElementById("myDiv");