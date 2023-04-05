from rdm.data.openimages_bbox_helpers.openimages_builderutils import Category

open_images_unify_categories_for_coco = {
    '/m/03bt1vf': '/m/01g317',
    '/m/04yx4': '/m/01g317',
    '/m/05r655': '/m/01g317',
    '/m/01bl7v': '/m/01g317',
    '/m/0cnyhnx': '/m/01xq0k1',
    '/m/01226z': '/m/018xm',
    '/m/05ctyq': '/m/018xm',
    '/m/058qzx': '/m/04ctx',
    '/m/06pcq': '/m/0l515',
    '/m/03m3pdh': '/m/02crq1',
    '/m/046dlr': '/m/01x3z',
    '/m/0h8mzrc': '/m/01x3z',
}

mixed_dataset_category_mapping = [
    [Category(id='1', super_category='person', name='person'),
     Category(id='/m/03bt1vf', super_category=None, name='Woman'),
     Category(id='/m/04yx4', super_category=None, name='Man'),
     Category(id='/m/05r655', super_category=None, name='Girl'),
     Category(id='/m/01bl7v', super_category=None, name='Boy'),
     Category(id='/m/01g317', super_category=None, name='Person')],

    [Category(id='2', super_category='vehicle', name='bicycle'),
     Category(id='/m/0199g', super_category=None, name='Bicycle')],

    [Category(id='3', super_category='vehicle', name='car'),
     Category(id='/m/0k4j', super_category=None, name='Car')],

    [Category(id='4', super_category='vehicle', name='motorcycle'),
     Category(id='/m/04_sv', super_category=None, name='Motorcycle')],

    [Category(id='5', super_category='vehicle', name='airplane'),
     Category(id='/m/0cmf2', super_category=None, name='Airplane')],

    [Category(id='6', super_category='vehicle', name='bus'),
     Category(id='/m/01bjv', super_category=None, name='Bus')],

    [Category(id='7', super_category='vehicle', name='train'),
     Category(id='/m/07jdr', super_category=None, name='Train')],

    [Category(id='8', super_category='vehicle', name='truck'),
     Category(id='/m/07r04', super_category=None, name='Truck')],

    [Category(id='9', super_category='vehicle', name='boat'),
     Category(id='/m/019jd', super_category=None, name='Boat')],

    [Category(id='10', super_category='outdoor', name='traffic light'),
     Category(id='/m/015qff', super_category=None, name='Traffic light')],

    [Category(id='11', super_category='outdoor', name='fire hydrant'),
     Category(id='/m/01pns0', super_category=None, name='Fire hydrant')],

    [Category(id='13', super_category='outdoor', name='stop sign'),
     Category(id='/m/02pv19', super_category=None, name='Stop sign')],

    [Category(id='14', super_category='outdoor', name='parking meter'),
     Category(id='/m/015qbp', super_category=None, name='Parking meter')],

    [Category(id='15', super_category='outdoor', name='bench'),
     Category(id='/m/0cvnqh', super_category=None, name='Bench')],

    [Category(id='16', super_category='animal', name='bird'),
     Category(id='/m/015p6', super_category=None, name='Bird')],

    [Category(id='17', super_category='animal', name='cat'),
     Category(id='/m/01yrx', super_category=None, name='Cat')],

    [Category(id='18', super_category='animal', name='dog'),
     Category(id='/m/0bt9lr', super_category=None, name='Dog')],

    [Category(id='19', super_category='animal', name='horse'),
     Category(id='/m/03k3r', super_category=None, name='Horse')],

    [Category(id='20', super_category='animal', name='sheep'),
     Category(id='/m/07bgp', super_category=None, name='Sheep')],

    [Category(id='21', super_category='animal', name='cow'),
     Category(id='/m/01xq0k1', super_category=None, name='Cattle'),
     Category(id='/m/0cnyhnx', super_category=None, name='Bull')],

    [Category(id='22', super_category='animal', name='elephant'),
     Category(id='/m/0bwd_0j', super_category=None, name='Elephant')],

    [Category(id='23', super_category='animal', name='bear'),
     Category(id='/m/01dws', super_category=None, name='Bear')],

    [Category(id='24', super_category='animal', name='zebra'),
     Category(id='/m/0898b', super_category=None, name='Zebra')],

    [Category(id='25', super_category='animal', name='giraffe'),
     Category(id='/m/03bk1', super_category=None, name='Giraffe')],

    [Category(id='27', super_category='accessory', name='backpack'),
     Category(id='/m/01940j', super_category=None, name='Backpack')],

    [Category(id='28', super_category='accessory', name='umbrella'),
     Category(id='/m/0hnnb', super_category=None, name='Umbrella')],

    [Category(id='31', super_category='accessory', name='handbag'),
     Category(id='/m/080hkjn', super_category=None, name='Handbag')],

    [Category(id='32', super_category='accessory', name='tie'),
     Category(id='/m/01rkbr', super_category=None, name='Tie')],

    [Category(id='33', super_category='accessory', name='suitcase'),
     Category(id='/m/01s55n', super_category=None, name='Suitcase')],

    [Category(id='34', super_category='sports', name='frisbee'),
     Category(id='/m/02wmf', super_category=None, name='Flying disc')],

    [Category(id='35', super_category='sports', name='skis'),
     Category(id='/m/071p9', super_category=None, name='Ski')],

    [Category(id='36', super_category='sports', name='snowboard'),
     Category(id='/m/06__v', super_category=None, name='Snowboard')],

    [Category(id='37', super_category='sports', name='sports ball'),
     Category(id='/m/01226z', super_category=None, name='Football'),
     Category(id='/m/018xm', super_category=None, name='Ball'),
     Category(id='/m/05ctyq', super_category=None, name='Tennis ball')],

    [Category(id='38', super_category='sports', name='kite'),
     Category(id='/m/02zt3', super_category=None, name='Kite')],

    [Category(id='39', super_category='sports', name='baseball bat'),
     Category(id='/m/03g8mr', super_category=None, name='Baseball bat')],

    [Category(id='40', super_category='sports', name='baseball glove'),
     Category(id='/m/03grzl', super_category=None, name='Baseball glove')],

    [Category(id='41', super_category='sports', name='skateboard'),
     Category(id='/m/06_fw', super_category=None, name='Skateboard')],

    [Category(id='42', super_category='sports', name='surfboard'),
     Category(id='/m/019w40', super_category=None, name='Surfboard')],

    [Category(id='43', super_category='sports', name='tennis racket'),
     Category(id='/m/0h8my_4', super_category=None, name='Tennis racket')],

    [Category(id='44', super_category='kitchen', name='bottle'),
     Category(id='/m/04dr76w', super_category=None, name='Bottle')],

    [Category(id='46', super_category='kitchen', name='wine glass'),
     Category(id='/m/09tvcd', super_category=None, name='Wine glass')],

    [Category(id='47', super_category='kitchen', name='cup'),
     Category(id='/m/02p5f1q', super_category=None, name='Coffee cup')],

    [Category(id='48', super_category='kitchen', name='fork'),
     Category(id='/m/0dt3t', super_category=None, name='Fork')],

    [Category(id='49', super_category='kitchen', name='knife'),
     Category(id='/m/058qzx', super_category=None, name='Kitchen knife'),
     Category(id='/m/04ctx', super_category=None, name='Knife')],

    [Category(id='50', super_category='kitchen', name='spoon'),
     Category(id='/m/0cmx8', super_category=None, name='Spoon')],

    [Category(id='51', super_category='kitchen', name='bowl'),
     Category(id='/m/04kkgm', super_category=None, name='Bowl')],

    [Category(id='52', super_category='food', name='banana'),
     Category(id='/m/09qck', super_category=None, name='Banana')],

    [Category(id='53', super_category='food', name='apple'),
     Category(id='/m/014j1m', super_category=None, name='Apple')],

    [Category(id='54', super_category='food', name='sandwich'),
     Category(id='/m/06pcq', super_category=None, name='Submarine sandwich'),
     Category(id='/m/0l515', super_category=None, name='Sandwich')],

    [Category(id='55', super_category='food', name='orange'),
     Category(id='/m/0cyhj_', super_category=None, name='Orange')],

    [Category(id='56', super_category='food', name='broccoli'),
     Category(id='/m/0hkxq', super_category=None, name='Broccoli')],

    [Category(id='57', super_category='food', name='carrot'),
     Category(id='/m/0fj52s', super_category=None, name='Carrot')],

    [Category(id='58', super_category='food', name='hot dog'),
     Category(id='/m/01b9xk', super_category=None, name='Hot dog')],

    [Category(id='59', super_category='food', name='pizza'),
     Category(id='/m/0663v', super_category=None, name='Pizza')],

    [Category(id='60', super_category='food', name='donut'),
     Category(id='/m/0jy4k', super_category=None, name='Doughnut')],

    [Category(id='61', super_category='food', name='cake'),
     Category(id='/m/0fszt', super_category=None, name='Cake')],

    [Category(id='62', super_category='furniture', name='chair'),
     Category(id='/m/01mzpv', super_category=None, name='Chair')],

    [Category(id='63', super_category='furniture', name='couch'),
     Category(id='/m/03m3pdh', super_category=None, name='Sofa bed'),
     Category(id='/m/02crq1', super_category=None, name='Couch')],

    [Category(id='64', super_category='furniture', name='potted plant'),
     Category(id='/m/03fp41', super_category=None, name='Houseplant')],

    [Category(id='65', super_category='furniture', name='bed'),
     Category(id='/m/03ssj5', super_category=None, name='Bed')],

    [Category(id='67', super_category='furniture', name='dining table'),
     Category(id='/m/04bcr3', super_category=None, name='Table')],

    [Category(id='70', super_category='furniture', name='toilet'),
     Category(id='/m/09g1w', super_category=None, name='Toilet')],

    [Category(id='72', super_category='electronic', name='tv'),
     Category(id='/m/07c52', super_category=None, name='Television')],

    [Category(id='73', super_category='electronic', name='laptop'),
     Category(id='/m/01c648', super_category=None, name='Laptop')],

    [Category(id='74', super_category='electronic', name='mouse'),
     Category(id='/m/020lf', super_category=None, name='Computer mouse')],

    [Category(id='75', super_category='electronic', name='remote'),
     Category(id='/m/0qjjc', super_category=None, name='Remote control')],

    [Category(id='76', super_category='electronic', name='keyboard'),
     Category(id='/m/01m2v', super_category=None, name='Computer keyboard')],

    [Category(id='77', super_category='electronic', name='cell phone'),
     Category(id='/m/050k8', super_category=None, name='Mobile phone')],

    [Category(id='78', super_category='appliance', name='microwave'),
     Category(id='/m/0fx9l', super_category=None, name='Microwave oven')],

    [Category(id='79', super_category='appliance', name='oven'),
     Category(id='/m/029bxz', super_category=None, name='Oven')],

    [Category(id='80', super_category='appliance', name='toaster'),
     Category(id='/m/01k6s3', super_category=None, name='Toaster')],

    [Category(id='81', super_category='appliance', name='sink'),
     Category(id='/m/0130jx', super_category=None, name='Sink')],

    [Category(id='82', super_category='appliance', name='refrigerator'),
     Category(id='/m/040b_t', super_category=None, name='Refrigerator')],

    [Category(id='84', super_category='indoor', name='book'),
     Category(id='/m/0bt_c3', super_category=None, name='Book')],

    [Category(id='85', super_category='indoor', name='clock'),
     Category(id='/m/01x3z', super_category=None, name='Clock'),
     Category(id='/m/046dlr', super_category=None, name='Alarm clock'),
     Category(id='/m/0h8mzrc', super_category=None, name='Wall clock')],

    [Category(id='86', super_category='indoor', name='vase'),
     Category(id='/m/02s195', super_category=None, name='Vase')],

    [Category(id='87', super_category='indoor', name='scissors'),
     Category(id='/m/01lsmm', super_category=None, name='Scissors')],

    [Category(id='88', super_category='indoor', name='teddy bear'),
     Category(id='/m/0kmg4', super_category=None, name='Teddy bear')],

    [Category(id='89', super_category='indoor', name='hair drier'),
     Category(id='/m/03wvsk', super_category=None, name='Hair dryer')],

    [Category(id='90', super_category='indoor', name='toothbrush'),
     Category(id='/m/012xff', super_category=None, name='Toothbrush')]
]

no_categories = len(mixed_dataset_category_mapping)
